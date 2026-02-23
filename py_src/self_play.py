import numpy as np
import torch
import torch.nn.functional as F
import time

import alphaclaude_cpp as ac
from .config import AlphaClaudeConfig
from .model import AlphaZeroNet


class SelfPlayWorker:
    def __init__(self, model: AlphaZeroNet, config: AlphaClaudeConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def play_game(self) -> list:
        """Play one self-play game, returning list of (nn_input, policy_target, value_target) tuples.
        Value targets are filled in at the end from the game outcome.
        Kept for single-game use (e.g. arena)."""
        self.model.eval()

        game = ac.GameState()
        mcts_config = ac.MCTSConfig()
        mcts_config.num_simulations = self.config.num_simulations
        mcts_config.c_base = self.config.c_base
        mcts_config.c_init = self.config.c_init
        mcts_config.dirichlet_alpha = self.config.dirichlet_alpha
        mcts_config.dirichlet_epsilon = self.config.dirichlet_epsilon
        mcts_config.batch_size = self.config.mcts_batch_size

        mcts = ac.MCTS(mcts_config)
        trajectory = []  # (nn_input, policy_target, side_to_move)

        move_num = 0
        while not game.is_terminal() and move_num < self.config.max_game_length:
            # Run MCTS search
            mcts.new_search(game)

            while not mcts.search_complete():
                inputs, masks, terminal_indices, terminal_values = mcts.get_leaf_batch()
                n = inputs.shape[0]

                if n > 0:
                    # NN inference
                    inp_tensor = torch.from_numpy(inputs).to(self.device)
                    mask_tensor = torch.from_numpy(masks).to(self.device)

                    policy_logits, values = self.model(inp_tensor)

                    # Mask illegal moves (set to -inf before softmax)
                    policy_logits = policy_logits.masked_fill(mask_tensor == 0, -1e9)

                    # Provide raw logits to MCTS (it does softmax internally)
                    mcts.provide_evaluations(
                        values.squeeze(-1).cpu().numpy(),
                        policy_logits.cpu().numpy()
                    )

            # Store training data
            nn_input = game.get_nn_input()
            temperature = self.config.temperature if move_num < self.config.temp_threshold else self.config.temperature_final
            policy_target = mcts.get_policy_target(temperature)

            trajectory.append((nn_input, policy_target, game.side_to_move()))

            # Select and play move
            move_uci = mcts.select_move_uci(temperature)
            game.make_move_uci(move_uci)
            move_num += 1

        # Determine game outcome
        if game.is_terminal():
            result = game.terminal_value()
            terminal_side = game.side_to_move()
        else:
            result = 0.0  # too long, treat as draw

        # Build training examples with correct value assignment
        examples = []
        for nn_input, policy_target, side in trajectory:
            if game.is_terminal() and result != 0.0:
                value = result if side == terminal_side else -result
            else:
                value = 0.0
            examples.append((nn_input, policy_target, np.float32(value)))

        return examples


def _finalize_trajectory(trajectory, game):
    """Convert a trajectory + finished game into training examples."""
    if game.is_terminal():
        result = game.terminal_value()
        terminal_side = game.side_to_move()
    else:
        result = 0.0

    examples = []
    for nn_input, policy_target, side in trajectory:
        if game.is_terminal() and result != 0.0:
            value = result if side == terminal_side else -result
        else:
            value = 0.0
        examples.append((nn_input, policy_target, np.float32(value)))
    return examples


class _GameSlot:
    """Lightweight Python-side state for one concurrent game."""
    __slots__ = ('game', 'trajectory', 'move_num', 'active')

    def __init__(self):
        self.game = ac.GameState()
        self.trajectory = []
        self.move_num = 0
        self.active = True


class ParallelSelfPlay:
    """Runs N games concurrently, batching NN inference across all active searches.
    Uses C++ ParallelMCTS to eliminate the Python loop over trees."""

    def __init__(self, model: AlphaZeroNet, config: AlphaClaudeConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.use_amp = (device.type == 'cuda')

    def _make_mcts_config(self):
        cfg = ac.MCTSConfig()
        cfg.num_simulations = self.config.num_simulations
        cfg.c_base = self.config.c_base
        cfg.c_init = self.config.c_init
        cfg.dirichlet_alpha = self.config.dirichlet_alpha
        cfg.dirichlet_epsilon = self.config.dirichlet_epsilon
        cfg.batch_size = self.config.mcts_batch_size
        return cfg

    @torch.no_grad()
    def run(self, num_games: int) -> list:
        """Play num_games self-play games with cross-game batched inference.
        Returns all training examples."""
        self.model.eval()
        mcts_config = self._make_mcts_config()
        n_parallel = min(self.config.num_parallel_games, num_games)

        # C++ side: one ParallelMCTS manages all trees
        pmcts = ac.ParallelMCTS(mcts_config, n_parallel)

        # Python side: lightweight game state + trajectory tracking
        slots = [_GameSlot() for _ in range(n_parallel)]
        for i, s in enumerate(slots):
            pmcts.new_search(i, s.game)

        all_examples = []
        games_started = n_parallel
        games_completed = 0
        start_time = time.perf_counter()
        gather_time = 0.0
        infer_time = 0.0
        provide_time = 0.0
        infer_calls = 0
        total_leaf_evals = 0
        max_leaf_batch = 0

        while games_completed < num_games:
            # --- ONE C++ call gathers leaves from ALL trees ---
            t0 = time.perf_counter()
            inputs, masks, game_ids, batch_counts = pmcts.get_all_leaf_batches()
            gather_time += time.perf_counter() - t0
            n = inputs.shape[0]

            # --- ONE GPU call for all leaves ---
            if n > 0:
                total_leaf_evals += int(n)
                infer_calls += 1
                if n > max_leaf_batch:
                    max_leaf_batch = int(n)

                inp_tensor = torch.from_numpy(inputs).to(self.device, non_blocking=True)
                mask_tensor = torch.from_numpy(masks).to(self.device, non_blocking=True)

                t1 = time.perf_counter()
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        policy_logits, values = self.model(inp_tensor)
                    policy_logits = policy_logits.float()
                    values = values.float()
                else:
                    policy_logits, values = self.model(inp_tensor)

                policy_logits = policy_logits.masked_fill(mask_tensor == 0, -1e9)

                values_np = values.squeeze(-1).cpu().numpy()
                policy_np = policy_logits.cpu().numpy()
                infer_time += time.perf_counter() - t1

                # --- ONE C++ call distributes results to all trees ---
                t2 = time.perf_counter()
                pmcts.provide_all_evaluations(
                    values_np, policy_np, game_ids, batch_counts
                )
                provide_time += time.perf_counter() - t2

            # --- Handle completed searches (lightweight Python) ---
            for i in range(n_parallel):
                if not slots[i].active:
                    continue
                if not pmcts.search_complete(i):
                    continue

                slot = slots[i]
                temp = (self.config.temperature
                        if slot.move_num < self.config.temp_threshold
                        else self.config.temperature_final)

                # Record training data
                nn_input = slot.game.get_nn_input()
                policy_target = pmcts.get_policy_target(i, temp)
                slot.trajectory.append((nn_input, policy_target, slot.game.side_to_move()))

                # Play the move
                move_uci = pmcts.select_move_uci(i, temp)
                slot.game.make_move_uci(move_uci)
                slot.move_num += 1

                # Check if game is over
                if slot.game.is_terminal() or slot.move_num >= self.config.max_game_length:
                    examples = _finalize_trajectory(slot.trajectory, slot.game)
                    all_examples.extend(examples)
                    games_completed += 1

                    if games_completed % 10 == 0:
                        print(f"  Self-play: {games_completed}/{num_games} games, "
                              f"{len(all_examples)} examples")

                    # Reuse slot for a new game
                    if games_started < num_games:
                        slot.game = ac.GameState()
                        slot.trajectory = []
                        slot.move_num = 0
                        pmcts.reset_game(i)
                        pmcts.new_search(i, slot.game)
                        games_started += 1
                    else:
                        slot.active = False
                else:
                    # Start next search for this game
                    pmcts.new_search(i, slot.game)

        if self.config.log_perf:
            total_time = time.perf_counter() - start_time
            examples_per_sec = len(all_examples) / total_time if total_time > 0 else 0.0
            games_per_sec = num_games / total_time if total_time > 0 else 0.0
            avg_leaf_batch = total_leaf_evals / infer_calls if infer_calls > 0 else 0.0
            infer_share = 100.0 * infer_time / total_time if total_time > 0 else 0.0
            gather_share = 100.0 * gather_time / total_time if total_time > 0 else 0.0
            provide_share = 100.0 * provide_time / total_time if total_time > 0 else 0.0
            print("  Self-play perf: "
                  f"{games_per_sec:.2f} games/s, {examples_per_sec:.1f} ex/s, "
                  f"infer_calls={infer_calls}, avg_leaf_batch={avg_leaf_batch:.1f}, "
                  f"max_leaf_batch={max_leaf_batch}, "
                  f"t_get={gather_share:.1f}%, t_nn={infer_share:.1f}%, "
                  f"t_provide={provide_share:.1f}%")

        return all_examples


def run_self_play(model: AlphaZeroNet, config: AlphaClaudeConfig, device: torch.device,
                  num_games: int = None) -> list:
    """Run multiple self-play games with cross-game batched inference."""
    if num_games is None:
        num_games = config.num_self_play_games

    parallel = ParallelSelfPlay(model, config, device)
    return parallel.run(num_games)
