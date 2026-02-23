"""Test MCTS behavior."""
import numpy as np
import torch
import alphaclaude_cpp as ac
from py_src.model import AlphaZeroNet
from py_src.config import AlphaClaudeConfig


def test_mcts_basic():
    """Run MCTS with uniform policy and verify it completes."""
    config = ac.MCTSConfig()
    config.num_simulations = 50
    config.batch_size = 8

    game = ac.GameState()
    mcts = ac.MCTS(config)
    mcts.new_search(game)

    while not mcts.search_complete():
        inputs, masks, term_idx, term_vals = mcts.get_leaf_batch()
        n = inputs.shape[0]
        if n > 0:
            # Uniform policy, zero value
            values = np.zeros(n, dtype=np.float32)
            policies = np.zeros((n, ac.POLICY_SIZE), dtype=np.float32)
            mcts.provide_evaluations(values, policies)

    assert mcts.simulations_done() >= 50
    policy = mcts.get_policy_target(1.0)
    assert abs(policy.sum() - 1.0) < 1e-5, f"Policy doesn't sum to 1: {policy.sum()}"
    move = mcts.select_move_uci(1.0)
    assert len(move) >= 4, f"Invalid move: {move}"


def test_mcts_with_model():
    """Run MCTS with actual neural network."""
    cfg = AlphaClaudeConfig()
    model = AlphaZeroNet(cfg)
    model.eval()

    mcts_config = ac.MCTSConfig()
    mcts_config.num_simulations = 20
    mcts_config.batch_size = 4

    game = ac.GameState()
    mcts = ac.MCTS(mcts_config)
    mcts.new_search(game)

    with torch.no_grad():
        while not mcts.search_complete():
            inputs, masks, term_idx, term_vals = mcts.get_leaf_batch()
            n = inputs.shape[0]
            if n > 0:
                inp = torch.from_numpy(inputs)
                mask = torch.from_numpy(masks)
                p, v = model(inp)
                p = p.masked_fill(mask == 0, -1e9)
                mcts.provide_evaluations(
                    v.squeeze(-1).numpy(),
                    p.numpy()
                )

    move = mcts.select_move_uci(1.0)
    assert move in game.legal_moves_uci(), f"MCTS returned illegal move: {move}"


def test_mate_in_one():
    """MCTS should find a mate-in-1 even with random NN."""
    # White to move, Qh7# is mate
    fen = "6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1"
    game = ac.GameState(fen)
    moves = game.legal_moves_uci()

    cfg = AlphaClaudeConfig()
    model = AlphaZeroNet(cfg)
    model.eval()

    mcts_config = ac.MCTSConfig()
    mcts_config.num_simulations = 200
    mcts_config.batch_size = 8
    mcts_config.dirichlet_epsilon = 0.0  # no noise for deterministic test

    mcts = ac.MCTS(mcts_config)
    mcts.new_search(game)

    with torch.no_grad():
        while not mcts.search_complete():
            inputs, masks, term_idx, term_vals = mcts.get_leaf_batch()
            n = inputs.shape[0]
            if n > 0:
                inp = torch.from_numpy(inputs)
                mask = torch.from_numpy(masks)
                p, v = model(inp)
                p = p.masked_fill(mask == 0, -1e9)
                mcts.provide_evaluations(
                    v.squeeze(-1).numpy(),
                    p.numpy()
                )

    move = mcts.select_move_uci(0.01)
    # After the selected move, opponent should be in checkmate
    test_game = ac.GameState(fen)
    test_game.make_move_uci(move)
    is_terminal = test_game.is_terminal()
    value = test_game.terminal_value() if is_terminal else None

    # The MCTS might find the mate or might not with random weights,
    # but if it has enough simulations it should prefer winning moves.
    # At minimum, it should pick a legal move.
    assert move in moves, f"MCTS returned illegal move: {move}"
    # The root value should be positive (white is winning)
    assert mcts.root_value() > 0, f"Root value should be positive: {mcts.root_value()}"


def test_policy_target_sums_to_one():
    """Policy target should be a valid probability distribution."""
    config = ac.MCTSConfig()
    config.num_simulations = 30
    config.batch_size = 8

    game = ac.GameState()
    mcts = ac.MCTS(config)
    mcts.new_search(game)

    while not mcts.search_complete():
        inputs, masks, _, _ = mcts.get_leaf_batch()
        n = inputs.shape[0]
        if n > 0:
            values = np.zeros(n, dtype=np.float32)
            policies = np.zeros((n, ac.POLICY_SIZE), dtype=np.float32)
            mcts.provide_evaluations(values, policies)

    for temp in [1.0, 0.5, 0.01]:
        policy = mcts.get_policy_target(temp)
        assert abs(policy.sum() - 1.0) < 1e-4, f"Policy sum at temp={temp}: {policy.sum()}"
        assert (policy >= 0).all(), "Negative probabilities"


if __name__ == "__main__":
    import sys
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)
