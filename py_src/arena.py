import numpy as np
import torch

import alphaclaude_cpp as ac
from .config import AlphaClaudeConfig
from .model import AlphaZeroNet


@torch.no_grad()
def play_arena_game(model1: AlphaZeroNet, model2: AlphaZeroNet,
                    config: AlphaClaudeConfig, device: torch.device,
                    model1_white: bool = True) -> float:
    """Play a single game between two models. Returns result from model1's perspective (+1, 0, -1)."""
    models = {0: model1 if model1_white else model2,
              1: model2 if model1_white else model1}

    for m in models.values():
        m.eval()

    game = ac.GameState()
    mcts_config = ac.MCTSConfig()
    mcts_config.num_simulations = config.num_simulations // 2  # faster for arena
    mcts_config.c_base = config.c_base
    mcts_config.c_init = config.c_init
    mcts_config.dirichlet_alpha = config.dirichlet_alpha
    mcts_config.dirichlet_epsilon = 0.0  # no noise in arena
    mcts_config.batch_size = config.mcts_batch_size

    mcts = ac.MCTS(mcts_config)
    move_num = 0

    while not game.is_terminal() and move_num < config.max_game_length:
        current_model = models[game.side_to_move()]

        mcts.new_search(game)
        while not mcts.search_complete():
            inputs, masks, terminal_indices, terminal_values = mcts.get_leaf_batch()
            n = inputs.shape[0]
            if n > 0:
                inp_tensor = torch.from_numpy(inputs).to(device)
                mask_tensor = torch.from_numpy(masks).to(device)
                policy_logits, values = current_model(inp_tensor)
                policy_logits = policy_logits.masked_fill(mask_tensor == 0, -1e9)
                mcts.provide_evaluations(
                    values.squeeze(-1).cpu().numpy(),
                    policy_logits.cpu().numpy()
                )

        # Use low temperature for arena (near-deterministic)
        move_uci = mcts.select_move_uci(0.01)
        game.make_move_uci(move_uci)
        move_num += 1

    if not game.is_terminal():
        return 0.0  # draw by max length

    result = game.terminal_value()  # from current player's perspective
    current_side = game.side_to_move()

    # Convert to model1's perspective
    # result is from current_side's perspective at terminal
    if result == 0.0:
        return 0.0

    # Determine which model is current_side
    model1_is_current = (current_side == 0 and model1_white) or (current_side == 1 and not model1_white)
    if model1_is_current:
        return result  # result is already from model1's perspective
    else:
        return -result


def pit(model1: AlphaZeroNet, model2: AlphaZeroNet,
        config: AlphaClaudeConfig, device: torch.device,
        num_games: int = None) -> float:
    """Play multiple games between two models, alternating colors.
    Returns model1's win rate (wins + 0.5*draws) / total."""
    if num_games is None:
        num_games = config.arena_games

    wins = 0
    draws = 0
    losses = 0

    for i in range(num_games):
        model1_white = (i % 2 == 0)
        result = play_arena_game(model1, model2, config, device, model1_white)

        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            total = wins + draws + losses
            wr = (wins + 0.5 * draws) / total
            print(f"  Arena: {i+1}/{num_games} - W:{wins} D:{draws} L:{losses} ({wr:.1%})")

    total = wins + draws + losses
    return (wins + 0.5 * draws) / total if total > 0 else 0.5
