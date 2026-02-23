import numpy as np
import torch
import torch.nn.functional as F

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
    """State for one concurrent game in parallel self-play."""
    __slots__ = ('game', 'mcts', 'trajectory', 'move_num', 'search_active')

    def __init__(self, mcts_config):
        self.game = ac.GameState()
        self.mcts = ac.MCTS(mcts_config)
        self.trajectory = []
        self.move_num = 0
        self.search_active = False


class ParallelSelfPlay:
    """Runs N games concurrently, batching NN inference across all active searches."""

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

        # Initialize game slots
        slots = [_GameSlot(mcts_config) for _ in range(n_parallel)]
        for s in slots:
            s.mcts.new_search(s.game)
            s.search_active = True

        all_examples = []
        games_started = n_parallel
        games_completed = 0

        while games_completed < num_games:
            # --- Gather leaf batches from all active searches ---
            slot_inputs = []     # list of (slot_idx, inputs_np, masks_np)
            all_inputs_list = []
            all_masks_list = []

            for idx, slot in enumerate(slots):
                if not slot.search_active:
                    continue
                inputs, masks, terminal_indices, terminal_values = slot.mcts.get_leaf_batch()
                n = inputs.shape[0]
                if n > 0:
                    slot_inputs.append((idx, n))
                    all_inputs_list.append(inputs)
                    all_masks_list.append(masks)

            # --- Batched NN inference ---
            if all_inputs_list:
                all_inp = np.concatenate(all_inputs_list, axis=0)
                all_msk = np.concatenate(all_masks_list, axis=0)

                inp_tensor = torch.from_numpy(all_inp).to(self.device)
                mask_tensor = torch.from_numpy(all_msk).to(self.device)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        policy_logits, values = self.model(inp_tensor)
                    # Cast back to float32 for numpy conversion
                    policy_logits = policy_logits.float()
                    values = values.float()
                else:
                    policy_logits, values = self.model(inp_tensor)

                policy_logits = policy_logits.masked_fill(mask_tensor == 0, -1e9)

                all_values_np = values.squeeze(-1).cpu().numpy()
                all_policy_np = policy_logits.cpu().numpy()

                # --- Distribute results back to each game's MCTS ---
                offset = 0
                for slot_idx, count in slot_inputs:
                    slot = slots[slot_idx]
                    slot.mcts.provide_evaluations(
                        all_values_np[offset:offset + count],
                        all_policy_np[offset:offset + count],
                    )
                    offset += count

            # --- Handle completed searches: record data, make moves ---
            for slot in slots:
                if not slot.search_active:
                    continue

                if slot.mcts.search_complete():
                    # Record training data for this position
                    temp = (self.config.temperature
                            if slot.move_num < self.config.temp_threshold
                            else self.config.temperature_final)
                    nn_input = slot.game.get_nn_input()
                    policy_target = slot.mcts.get_policy_target(temp)
                    slot.trajectory.append((nn_input, policy_target, slot.game.side_to_move()))

                    # Play the move
                    move_uci = slot.mcts.select_move_uci(temp)
                    slot.game.make_move_uci(move_uci)
                    slot.move_num += 1

                    # Check if game is over
                    if slot.game.is_terminal() or slot.move_num >= self.config.max_game_length:
                        # Finalize this game
                        examples = _finalize_trajectory(slot.trajectory, slot.game)
                        all_examples.extend(examples)
                        games_completed += 1

                        if (games_completed) % 10 == 0:
                            print(f"  Self-play: {games_completed}/{num_games} games, "
                                  f"{len(all_examples)} examples")

                        # Reuse slot for a new game if more games needed
                        if games_started < num_games:
                            slot.game = ac.GameState()
                            slot.mcts = ac.MCTS(mcts_config)
                            slot.trajectory = []
                            slot.move_num = 0
                            slot.mcts.new_search(slot.game)
                            slot.search_active = True
                            games_started += 1
                        else:
                            slot.search_active = False
                    else:
                        # Start next search for this game
                        slot.mcts.new_search(slot.game)

        return all_examples


def run_self_play(model: AlphaZeroNet, config: AlphaClaudeConfig, device: torch.device,
                  num_games: int = None) -> list:
    """Run multiple self-play games with cross-game batched inference."""
    if num_games is None:
        num_games = config.num_self_play_games

    parallel = ParallelSelfPlay(model, config, device)
    return parallel.run(num_games)
