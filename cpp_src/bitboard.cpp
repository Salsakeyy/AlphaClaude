#include "bitboard.h"
#include "magic.h"

namespace alphaclaude {

Bitboard KnightAttacks[SQUARE_NB];
Bitboard KingAttacks[SQUARE_NB];
Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard PseudoBishopAttacks[SQUARE_NB];
Bitboard PseudoRookAttacks[SQUARE_NB];

void bitboard_init() {
    // Initialize magic bitboards first (needed for line/between tables)
    magic_init();

    // Knight attacks
    const int knight_deltas[][2] = {
        {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
    };
    for (int sq = 0; sq < 64; sq++) {
        KnightAttacks[sq] = 0;
        int r = sq / 8, f = sq % 8;
        for (auto& d : knight_deltas) {
            int rr = r + d[0], ff = f + d[1];
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8)
                KnightAttacks[sq] |= 1ULL << (rr * 8 + ff);
        }
    }

    // King attacks
    const int king_deltas[][2] = {
        {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}
    };
    for (int sq = 0; sq < 64; sq++) {
        KingAttacks[sq] = 0;
        int r = sq / 8, f = sq % 8;
        for (auto& d : king_deltas) {
            int rr = r + d[0], ff = f + d[1];
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8)
                KingAttacks[sq] |= 1ULL << (rr * 8 + ff);
        }
    }

    // Pawn attacks
    for (int sq = 0; sq < 64; sq++) {
        Bitboard bb = square_bb(Square(sq));
        PawnAttacks[WHITE][sq] = shift<NORTH_WEST>(bb) | shift<NORTH_EAST>(bb);
        PawnAttacks[BLACK][sq] = shift<SOUTH_WEST>(bb) | shift<SOUTH_EAST>(bb);
    }

    // Pseudo-attacks on empty board
    for (int sq = 0; sq < 64; sq++) {
        PseudoBishopAttacks[sq] = bishop_attacks(Square(sq), 0);
        PseudoRookAttacks[sq] = rook_attacks(Square(sq), 0);
    }

    // Between and Line tables
    for (int s1 = 0; s1 < 64; s1++) {
        for (int s2 = 0; s2 < 64; s2++) {
            BetweenBB[s1][s2] = 0;
            LineBB[s1][s2] = 0;

            if (s1 == s2) continue;

            // Check if s2 is on a rook line from s1
            if (PseudoRookAttacks[s1] & square_bb(Square(s2))) {
                LineBB[s1][s2] = (rook_attacks(Square(s1), 0) & rook_attacks(Square(s2), 0))
                               | square_bb(Square(s1)) | square_bb(Square(s2));
                BetweenBB[s1][s2] = rook_attacks(Square(s1), square_bb(Square(s2)))
                                  & rook_attacks(Square(s2), square_bb(Square(s1)));
            }
            // Check if s2 is on a bishop line from s1
            else if (PseudoBishopAttacks[s1] & square_bb(Square(s2))) {
                LineBB[s1][s2] = (bishop_attacks(Square(s1), 0) & bishop_attacks(Square(s2), 0))
                               | square_bb(Square(s1)) | square_bb(Square(s2));
                BetweenBB[s1][s2] = bishop_attacks(Square(s1), square_bb(Square(s2)))
                                  & bishop_attacks(Square(s2), square_bb(Square(s1)));
            }
        }
    }
}

} // namespace alphaclaude
