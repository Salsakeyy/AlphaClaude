#pragma once

#include "types.h"

namespace alphaclaude {

// ============================================================
// Bitboard constants
// ============================================================

constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileBBB = FileABB << 1;
constexpr Bitboard FileCBB = FileABB << 2;
constexpr Bitboard FileDBB = FileABB << 3;
constexpr Bitboard FileEBB = FileABB << 4;
constexpr Bitboard FileFBB = FileABB << 5;
constexpr Bitboard FileGBB = FileABB << 6;
constexpr Bitboard FileHBB = FileABB << 7;

constexpr Bitboard Rank1BB = 0xFFULL;
constexpr Bitboard Rank2BB = Rank1BB << 8;
constexpr Bitboard Rank3BB = Rank1BB << 16;
constexpr Bitboard Rank4BB = Rank1BB << 24;
constexpr Bitboard Rank5BB = Rank1BB << 32;
constexpr Bitboard Rank6BB = Rank1BB << 40;
constexpr Bitboard Rank7BB = Rank1BB << 48;
constexpr Bitboard Rank8BB = Rank1BB << 56;

constexpr Bitboard FileBB[FILE_NB] = {
    FileABB, FileBBB, FileCBB, FileDBB, FileEBB, FileFBB, FileGBB, FileHBB
};

constexpr Bitboard RankBB[RANK_NB] = {
    Rank1BB, Rank2BB, Rank3BB, Rank4BB, Rank5BB, Rank6BB, Rank7BB, Rank8BB
};

constexpr Bitboard square_bb(Square s) { return 1ULL << s; }

// ============================================================
// Bit manipulation
// ============================================================

inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline Square lsb(Bitboard b) { return Square(__builtin_ctzll(b)); }
inline Square msb(Bitboard b) { return Square(63 ^ __builtin_clzll(b)); }

inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

inline bool more_than_one(Bitboard b) { return b & (b - 1); }

// ============================================================
// Shift operations (with wrapping prevention)
// ============================================================

template<Direction D>
constexpr Bitboard shift(Bitboard b) {
    if constexpr (D == NORTH)      return b << 8;
    if constexpr (D == SOUTH)      return b >> 8;
    if constexpr (D == EAST)       return (b & ~FileHBB) << 1;
    if constexpr (D == WEST)       return (b & ~FileABB) >> 1;
    if constexpr (D == NORTH_EAST) return (b & ~FileHBB) << 9;
    if constexpr (D == NORTH_WEST) return (b & ~FileABB) << 7;
    if constexpr (D == SOUTH_EAST) return (b & ~FileHBB) >> 7;
    if constexpr (D == SOUTH_WEST) return (b & ~FileABB) >> 9;
    return 0;
}

// ============================================================
// Attack tables (initialized at startup)
// ============================================================

extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];

// Between and Line tables for pin/check detection
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];  // squares strictly between s1 and s2
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];      // full line through s1 and s2

// Pseudo-attacks for bishops/rooks (on empty board, used for line/between computation)
extern Bitboard PseudoBishopAttacks[SQUARE_NB];
extern Bitboard PseudoRookAttacks[SQUARE_NB];

// ============================================================
// Magic bitboard interface (implemented in magic.cpp)
// ============================================================

Bitboard bishop_attacks(Square s, Bitboard occ);
Bitboard rook_attacks(Square s, Bitboard occ);

inline Bitboard queen_attacks(Square s, Bitboard occ) {
    return bishop_attacks(s, occ) | rook_attacks(s, occ);
}

// ============================================================
// Initialization
// ============================================================

void bitboard_init();

// ============================================================
// Utility: aligned squares for pin detection
// ============================================================

inline bool aligned(Square s1, Square s2, Square s3) {
    return LineBB[s1][s2] & square_bb(s3);
}

} // namespace alphaclaude
