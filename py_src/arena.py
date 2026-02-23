import numpy as np
import torch

import alphaclaude_cpp as ac
from .config import AlphaClaudeConfig
from .model import AlphaZeroNet


class _ArenaSlot:
    """Lightweight Python-side state for one concurrent arena game."""
    __slots__ = ('game', 'move_num', 'active', 'model1_white')

    def __init__(self, model1_white):
        self.game = ac.GameState()
        self.move_num = 0
        self.active = True
        self.model1_white = model1_white

    def is_model1_turn(self):
        side = self.game.side_to_move()
        if self.model1_white:
            return side == 0
        else:
            return side == 1


class ParallelArena:
    """Runs N arena games concurrently with cross-game batched inference.
    Uses C++ ParallelMCTS to eliminate the Python loop over trees."""

    def __init__(self, model1: AlphaZeroNet, model2: AlphaZeroNet,
                 config: AlphaClaudeConfig, device: torch.device):
        self.model1 = model1
        self.model2 = model2
        self.config = config
        self.device = device
        self.use_amp = (device.type == 'cuda')

    def _make_mcts_config(self):
        cfg = ac.MCTSConfig()
        cfg.num_simulations = self.config.num_simulations // 2  # faster for arena
        cfg.c_base = self.config.c_base
        cfg.c_init = self.config.c_init
        cfg.dirichlet_alpha = self.config.dirichlet_alpha
        cfg.dirichlet_epsilon = 0.0  # no noise in arena
        cfg.batch_size = self.config.mcts_batch_size
        return cfg

    def _result_for_model1(self, slot):
        """Get game result from model1's perspective."""
        if not slot.game.is_terminal():
            return 0.0
        result = slot.game.terminal_value()
        if result == 0.0:
            return 0.0
        current_side = slot.game.side_to_move()
        model1_is_current = ((current_side == 0 and slot.model1_white) or
                             (current_side == 1 and not slot.model1_white))
        return result if model1_is_current else -result

    @torch.no_grad()
    def run(self, num_games: int) -> float:
        """Play num_games arena games. Returns model1 win rate."""
        self.model1.eval()
        self.model2.eval()

        mcts_config = self._make_mcts_config()
        n_parallel = min(self.config.num_parallel_games, num_games)

        # Single ParallelMCTS for all games (trees are model-agnostic)
        pmcts = ac.ParallelMCTS(mcts_config, n_parallel)

        slots = []
        for i in range(n_parallel):
            slot = _ArenaSlot(model1_white=(i % 2 == 0))
            pmcts.new_search(i, slot.game)
            slots.append(slot)

        wins = 0
        draws = 0
        losses = 0
        games_started = n_parallel
        games_completed = 0

        while games_completed < num_games:
            # ONE C++ call gathers leaves from all trees
            inputs, masks, game_ids, batch_counts = pmcts.get_all_leaf_batches()
            n = inputs.shape[0]

            if n > 0:
                # Split leaves by which model should evaluate them
                m1_mask_idx = []  # indices into the flat [N,...] arrays for model1
                m2_mask_idx = []

                offset = 0
                for k in range(len(game_ids)):
                    gid = int(game_ids[k])
                    count = int(batch_counts[k])
                    if slots[gid].is_model1_turn():
                        m1_mask_idx.extend(range(offset, offset + count))
                    else:
                        m2_mask_idx.extend(range(offset, offset + count))
                    offset += count

                # Allocate combined output arrays
                all_values_np = np.empty(n, dtype=np.float32)
                all_policy_np = np.empty_like(masks)

                # Model1 inference
                if m1_mask_idx:
                    self._infer_subset(self.model1, inputs, masks,
                                       m1_mask_idx, all_values_np, all_policy_np)
                # Model2 inference
                if m2_mask_idx:
                    self._infer_subset(self.model2, inputs, masks,
                                       m2_mask_idx, all_values_np, all_policy_np)

                # ONE C++ call distributes results to all trees
                pmcts.provide_all_evaluations(
                    all_values_np, all_policy_np, game_ids, batch_counts
                )

            # Handle completed searches
            for i in range(n_parallel):
                if not slots[i].active:
                    continue
                if not pmcts.search_complete(i):
                    continue

                slot = slots[i]
                move_uci = pmcts.select_move_uci(i, 0.01)
                slot.game.make_move_uci(move_uci)
                slot.move_num += 1

                if slot.game.is_terminal() or slot.move_num >= self.config.max_game_length:
                    result = self._result_for_model1(slot)
                    if result > 0:
                        wins += 1
                    elif result < 0:
                        losses += 1
                    else:
                        draws += 1
                    games_completed += 1

                    if games_completed % 10 == 0:
                        total = wins + draws + losses
                        wr = (wins + 0.5 * draws) / total
                        print(f"  Arena: {games_completed}/{num_games} - "
                              f"W:{wins} D:{draws} L:{losses} ({wr:.1%})")

                    # Reuse slot
                    if games_started < num_games:
                        slot.game = ac.GameState()
                        slot.move_num = 0
                        slot.model1_white = (games_started % 2 == 0)
                        pmcts.reset_game(i)
                        pmcts.new_search(i, slot.game)
                        games_started += 1
                    else:
                        slot.active = False
                else:
                    pmcts.new_search(i, slot.game)

        total = wins + draws + losses
        return (wins + 0.5 * draws) / total if total > 0 else 0.5

    def _infer_subset(self, model, all_inputs, all_masks, indices,
                      out_values, out_policies):
        """Run inference on a subset of the batch and write results back."""
        idx = np.array(indices, dtype=np.int64)
        inp_tensor = torch.from_numpy(all_inputs[idx]).to(self.device)
        mask_tensor = torch.from_numpy(all_masks[idx]).to(self.device)

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                policy_logits, values = model(inp_tensor)
            policy_logits = policy_logits.float()
            values = values.float()
        else:
            policy_logits, values = model(inp_tensor)

        policy_logits = policy_logits.masked_fill(mask_tensor == 0, -1e9)

        vals_np = values.squeeze(-1).cpu().numpy()
        pols_np = policy_logits.cpu().numpy()

        out_values[idx] = vals_np
        out_policies[idx] = pols_np


def pit(model1: AlphaZeroNet, model2: AlphaZeroNet,
        config: AlphaClaudeConfig, device: torch.device,
        num_games: int = None) -> float:
    """Play multiple games between two models, alternating colors.
    Returns model1's win rate (wins + 0.5*draws) / total."""
    if num_games is None:
        num_games = config.arena_games

    arena = ParallelArena(model1, model2, config, device)
    return arena.run(num_games)
