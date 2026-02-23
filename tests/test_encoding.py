"""Test move encoding round-trips."""
import alphaclaude_cpp as ac


def test_startpos_encoding_roundtrip():
    g = ac.GameState()
    moves = g.legal_moves_uci()
    for m in moves:
        idx = g.encode_move(m)
        decoded = ac.GameState.decode_move_uci(idx, g.side_to_move())
        assert decoded == m, f"Failed: {m} -> idx={idx} -> {decoded}"


def test_kiwipete_encoding_roundtrip():
    g = ac.GameState("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    moves = g.legal_moves_uci()
    for m in moves:
        idx = g.encode_move(m)
        assert 0 <= idx < ac.POLICY_SIZE, f"Index out of range: {m} -> {idx}"
        decoded = ac.GameState.decode_move_uci(idx, g.side_to_move())
        assert decoded == m, f"Failed: {m} -> idx={idx} -> {decoded}"


def test_black_encoding_roundtrip():
    g = ac.GameState()
    g.make_move_uci("e2e4")
    moves = g.legal_moves_uci()
    for m in moves:
        idx = g.encode_move(m)
        assert 0 <= idx < ac.POLICY_SIZE, f"Index out of range: {m} -> {idx}"
        decoded = ac.GameState.decode_move_uci(idx, g.side_to_move())
        assert decoded == m, f"Failed: {m} -> idx={idx} -> {decoded}"


def test_promotion_encoding():
    g = ac.GameState("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    moves = g.legal_moves_uci()
    promo_moves = [m for m in moves if len(m) == 5]
    assert len(promo_moves) >= 4, f"Expected promotions, got: {promo_moves}"

    for m in promo_moves:
        idx = g.encode_move(m)
        assert 0 <= idx < ac.POLICY_SIZE, f"Index out of range: {m} -> {idx}"
        decoded = ac.GameState.decode_move_uci(idx, g.side_to_move())
        assert decoded == m, f"Failed: {m} -> idx={idx} -> {decoded}"


def test_legal_mask_matches_legal_moves():
    g = ac.GameState()
    mask = g.get_legal_move_mask()
    moves = g.legal_moves_uci()

    # Number of 1s in mask should equal number of legal moves
    num_ones = int(mask.sum())
    assert num_ones == len(moves), f"Mask has {num_ones} ones but {len(moves)} legal moves"


def test_all_indices_unique():
    g = ac.GameState("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    moves = g.legal_moves_uci()
    indices = [g.encode_move(m) for m in moves]
    assert len(set(indices)) == len(indices), "Duplicate move indices found"


if __name__ == "__main__":
    import sys
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)
