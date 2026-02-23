import numpy as np
import torch

import alphaclaude_cpp as ac
from .config import AlphaClaudeConfig
from .model import AlphaZeroNet


class _ArenaSlot:
    """State for one concurrent arena game."""
    __slots__ = ('game', 'mcts', 'move_num', 'search_active',
                 'model1_white', 'game_index')

    def __init__(self, mcts_config, model1_white, game_index):
        self.game = ac.GameState()
        self.mcts = ac.MCTS(mcts_config)
        self.move_num = 0
        self.search_active = False
        self.model1_white = model1_white
        self.game_index = game_index


class ParallelArena:
    """Runs N arena games concurrently with cross-game batched inference."""

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

    def _current_model(self, slot):
        """Return which model plays for the current side_to_move."""
        side = slot.game.side_to_move()
        # side 0 = white, side 1 = black
        if slot.model1_white:
            return self.model1 if side == 0 else self.model2
        else:
            return self.model1 if side == 1 else self.model2

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

        # Initialize game slots
        slots = []
        for i in range(n_parallel):
            slot = _ArenaSlot(mcts_config, model1_white=(i % 2 == 0), game_index=i)
            slot.mcts.new_search(slot.game)
            slot.search_active = True
            slots.append(slot)

        wins = 0
        draws = 0
        losses = 0
        games_started = n_parallel
        games_completed = 0

        while games_completed < num_games:
            # Group active slots by which model is currently playing
            # so we batch inference per model
            model1_batch = []  # (slot_idx, count)
            model2_batch = []
            m1_inputs = []
            m1_masks = []
            m2_inputs = []
            m2_masks = []

            for idx, slot in enumerate(slots):
                if not slot.search_active:
                    continue
                inputs, masks, terminal_indices, terminal_values = slot.mcts.get_leaf_batch()
                n = inputs.shape[0]
                if n > 0:
                    current_model = self._current_model(slot)
                    if current_model is self.model1:
                        model1_batch.append((idx, n))
                        m1_inputs.append(inputs)
                        m1_masks.append(masks)
                    else:
                        model2_batch.append((idx, n))
                        m2_inputs.append(inputs)
                        m2_masks.append(masks)

            # Inference for model1's batch
            if m1_inputs:
                self._batch_inference(self.model1, m1_inputs, m1_masks, model1_batch, slots)

            # Inference for model2's batch
            if m2_inputs:
                self._batch_inference(self.model2, m2_inputs, m2_masks, model2_batch, slots)

            # Handle completed searches
            for slot in slots:
                if not slot.search_active:
                    continue

                if slot.mcts.search_complete():
                    move_uci = slot.mcts.select_move_uci(0.01)
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
                            slot.mcts = ac.MCTS(mcts_config)
                            slot.move_num = 0
                            slot.model1_white = (games_started % 2 == 0)
                            slot.game_index = games_started
                            slot.mcts.new_search(slot.game)
                            slot.search_active = True
                            games_started += 1
                        else:
                            slot.search_active = False
                    else:
                        slot.mcts.new_search(slot.game)

        total = wins + draws + losses
        return (wins + 0.5 * draws) / total if total > 0 else 0.5

    def _batch_inference(self, model, inputs_list, masks_list, batch_info, slots):
        """Run batched inference for one model and distribute results."""
        all_inp = np.concatenate(inputs_list, axis=0)
        all_msk = np.concatenate(masks_list, axis=0)

        inp_tensor = torch.from_numpy(all_inp).to(self.device)
        mask_tensor = torch.from_numpy(all_msk).to(self.device)

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                policy_logits, values = model(inp_tensor)
            policy_logits = policy_logits.float()
            values = values.float()
        else:
            policy_logits, values = model(inp_tensor)

        policy_logits = policy_logits.masked_fill(mask_tensor == 0, -1e9)

        all_values_np = values.squeeze(-1).cpu().numpy()
        all_policy_np = policy_logits.cpu().numpy()

        offset = 0
        for slot_idx, count in batch_info:
            slots[slot_idx].mcts.provide_evaluations(
                all_values_np[offset:offset + count],
                all_policy_np[offset:offset + count],
            )
            offset += count


def pit(model1: AlphaZeroNet, model2: AlphaZeroNet,
        config: AlphaClaudeConfig, device: torch.device,
        num_games: int = None) -> float:
    """Play multiple games between two models, alternating colors.
    Returns model1's win rate (wins + 0.5*draws) / total."""
    if num_games is None:
        num_games = config.arena_games

    arena = ParallelArena(model1, model2, config, device)
    return arena.run(num_games)
