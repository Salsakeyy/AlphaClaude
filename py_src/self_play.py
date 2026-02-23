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
        Value targets are filled in at the end from the game outcome."""
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
            # result is from the perspective of the player whose turn it is at terminal state
            # We need to assign values from the perspective of the player at each step
            terminal_side = game.side_to_move()
        else:
            result = 0.0  # too long, treat as draw

        # Build training examples with correct value assignment
        examples = []
        for nn_input, policy_target, side in trajectory:
            if game.is_terminal() and result != 0.0:
                # result is from terminal_side's perspective
                # If this position's side == terminal_side, value = result
                # Otherwise, value = -result
                value = result if side == terminal_side else -result
            else:
                value = 0.0
            examples.append((nn_input, policy_target, np.float32(value)))

        return examples


def run_self_play(model: AlphaZeroNet, config: AlphaClaudeConfig, device: torch.device,
                  num_games: int = None) -> list:
    """Run multiple self-play games and return all training examples."""
    if num_games is None:
        num_games = config.num_self_play_games

    worker = SelfPlayWorker(model, config, device)
    all_examples = []

    for i in range(num_games):
        examples = worker.play_game()
        all_examples.extend(examples)
        if (i + 1) % 10 == 0:
            print(f"  Self-play: {i+1}/{num_games} games, {len(all_examples)} examples")

    return all_examples
