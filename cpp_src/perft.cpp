// Standalone perft test to validate the chess engine
#include "position.h"
#include "movegen.h"
#include <iostream>
#include <cstdint>

using namespace alphaclaude;

uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;

    MoveList moves;
    generate_legal(pos, moves);

    if (depth == 1) return moves.size();

    uint64_t nodes = 0;
    for (int i = 0; i < moves.size(); i++) {
        StateInfo si;
        pos.do_move(moves[i], si);
        nodes += perft(pos, depth - 1);
        pos.undo_move(moves[i]);
    }
    return nodes;
}

struct PerftTest {
    const char* fen;
    int depth;
    uint64_t expected;
};

int main() {
    PerftTest tests[] = {
        // Starting position
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},

        // Kiwipete
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},

        // Position 3: endgame
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},

        // Position 4: lots of promotions and captures
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},

        // Position 5
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
    };

    int pass = 0, fail = 0;
    int total = sizeof(tests) / sizeof(tests[0]);

    for (int i = 0; i < total; i++) {
        auto& t = tests[i];
        Position pos(t.fen);
        uint64_t result = perft(pos, t.depth);

        bool ok = (result == t.expected);
        if (ok) {
            pass++;
        } else {
            fail++;
            std::cout << "FAIL: " << t.fen << " depth " << t.depth
                      << " expected " << t.expected << " got " << result << std::endl;
        }
    }

    std::cout << "\nPerft results: " << pass << "/" << total << " passed";
    if (fail > 0) std::cout << ", " << fail << " FAILED";
    std::cout << std::endl;

    return fail > 0 ? 1 : 0;
}
