"""Perft tests to validate the chess engine."""
import alphaclaude_cpp as ac


def perft(game, depth):
    if depth == 0:
        return 1
    moves = game.legal_moves_uci()
    if depth == 1:
        return len(moves)
    nodes = 0
    for m in moves:
        g = ac.GameState(game.fen())
        g.make_move_uci(m)
        nodes += perft(g, depth - 1)
    return nodes


def test_startpos_depth1():
    g = ac.GameState()
    assert perft(g, 1) == 20

def test_startpos_depth2():
    g = ac.GameState()
    assert perft(g, 2) == 400

def test_startpos_depth3():
    g = ac.GameState()
    assert perft(g, 3) == 8902

def test_startpos_depth4():
    g = ac.GameState()
    assert perft(g, 4) == 197281

def test_kiwipete_depth1():
    g = ac.GameState("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert perft(g, 1) == 48

def test_kiwipete_depth2():
    g = ac.GameState("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert perft(g, 2) == 2039

def test_kiwipete_depth3():
    g = ac.GameState("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert perft(g, 3) == 97862

def test_endgame_depth4():
    g = ac.GameState("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    assert perft(g, 4) == 43238

def test_promotion_depth3():
    g = ac.GameState("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    assert perft(g, 3) == 9467

def test_position5_depth3():
    g = ac.GameState("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")
    assert perft(g, 3) == 62379


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
