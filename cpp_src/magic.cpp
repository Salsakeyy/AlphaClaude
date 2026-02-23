#include "magic.h"
#include "bitboard.h"

namespace alphaclaude {

Magic BishopMagics[64];
Magic RookMagics[64];

Bitboard BishopTable[0x1480];
Bitboard RookTable[0x19000];

// Hardcoded magic numbers (from well-known sources)
// These are proven to work with the given shift values.

static const Bitboard RookMagicNumbers[64] = {
    0x0080001020400080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL, 0x0080040800100080ULL,
    0x0080020400080080ULL, 0x0080010200040080ULL, 0x0080008001000200ULL, 0x0080002040800100ULL,
    0x0000800020400080ULL, 0x0000400020005000ULL, 0x0000801000200080ULL, 0x0000800800100080ULL,
    0x0000800400080080ULL, 0x0000800200040080ULL, 0x0000800100020080ULL, 0x0000800040800100ULL,
    0x0000208000400080ULL, 0x0000404000201000ULL, 0x0000808010002000ULL, 0x0000808008001000ULL,
    0x0000808004000800ULL, 0x0000808002000400ULL, 0x0000010100020004ULL, 0x0000020000408104ULL,
    0x0000208080004000ULL, 0x0000200040005000ULL, 0x0000100080200080ULL, 0x0000080080100080ULL,
    0x0000040080080080ULL, 0x0000020080040080ULL, 0x0000010080800200ULL, 0x0000800080004100ULL,
    0x0000204000800080ULL, 0x0000200040401000ULL, 0x0000100080802000ULL, 0x0000080080801000ULL,
    0x0000040080800800ULL, 0x0000020080800400ULL, 0x0000020001010004ULL, 0x0000800040800100ULL,
    0x0000204000808000ULL, 0x0000200040008080ULL, 0x0000100020008080ULL, 0x0000080010008080ULL,
    0x0000040008008080ULL, 0x0000020004008080ULL, 0x0000010002008080ULL, 0x0000004081020004ULL,
    0x0000204000800080ULL, 0x0000200040008080ULL, 0x0000100020008080ULL, 0x0000080010008080ULL,
    0x0000040008008080ULL, 0x0000020004008080ULL, 0x0000800100020080ULL, 0x0000800041000080ULL,
    0x00FFFCDDFCED714AULL, 0x007FFCDDFCED714AULL, 0x003FFFCDFFD88096ULL, 0x0000040810002101ULL,
    0x0001000204080011ULL, 0x0001000204000801ULL, 0x0001000082000401ULL, 0x0001FFFAABFAD1A2ULL
};

static const Bitboard BishopMagicNumbers[64] = {
    0x0002020202020200ULL, 0x0002020202020000ULL, 0x0004010202000000ULL, 0x0004040080000000ULL,
    0x0001104000000000ULL, 0x0000821040000000ULL, 0x0000410410400000ULL, 0x0000104104104000ULL,
    0x0000040404040400ULL, 0x0000020202020200ULL, 0x0000040102020000ULL, 0x0000040400800000ULL,
    0x0000011040000000ULL, 0x0000008210400000ULL, 0x0000004104104000ULL, 0x0000002082082000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800802004000ULL,
    0x0000800400A00000ULL, 0x0000200100884000ULL, 0x0000400082082000ULL, 0x0000200041041000ULL,
    0x0002080010101000ULL, 0x0001040008080800ULL, 0x0000208004010400ULL, 0x0000404004010200ULL,
    0x0000840000802000ULL, 0x0000404002011000ULL, 0x0000808001041000ULL, 0x0000404000820800ULL,
    0x0001041000202000ULL, 0x0000820800101000ULL, 0x0000104400080800ULL, 0x0000020080080080ULL,
    0x0000404040040100ULL, 0x0000808100020100ULL, 0x0001010100020800ULL, 0x0000808080010400ULL,
    0x0000820820004000ULL, 0x0000410410002000ULL, 0x0000082088001000ULL, 0x0000002011000800ULL,
    0x0000080100400400ULL, 0x0001010101000200ULL, 0x0002020202000400ULL, 0x0001010101000200ULL,
    0x0000410410400000ULL, 0x0000208208200000ULL, 0x0000002084100000ULL, 0x0000000020880000ULL,
    0x0000001002020000ULL, 0x0000040408020000ULL, 0x0004040404040000ULL, 0x0002020202020000ULL,
    0x0000104104104000ULL, 0x0000002082082000ULL, 0x0000000020841000ULL, 0x0000000000208800ULL,
    0x0000000010020200ULL, 0x0000000404080200ULL, 0x0000040404040400ULL, 0x0002020202020200ULL
};

static const int RookShifts[64] = {
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52
};

static const int BishopShifts[64] = {
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
};

// Compute rook attacks for a given square and occupancy (sliding ray approach)
static Bitboard compute_rook_attacks(Square sq, Bitboard occ) {
    Bitboard attacks = 0;
    int r = rank_of(sq), f = file_of(sq);

    // North
    for (int rr = r + 1; rr <= 7; rr++) {
        Bitboard b = 1ULL << (rr * 8 + f);
        attacks |= b;
        if (occ & b) break;
    }
    // South
    for (int rr = r - 1; rr >= 0; rr--) {
        Bitboard b = 1ULL << (rr * 8 + f);
        attacks |= b;
        if (occ & b) break;
    }
    // East
    for (int ff = f + 1; ff <= 7; ff++) {
        Bitboard b = 1ULL << (r * 8 + ff);
        attacks |= b;
        if (occ & b) break;
    }
    // West
    for (int ff = f - 1; ff >= 0; ff--) {
        Bitboard b = 1ULL << (r * 8 + ff);
        attacks |= b;
        if (occ & b) break;
    }
    return attacks;
}

// Compute bishop attacks for a given square and occupancy
static Bitboard compute_bishop_attacks(Square sq, Bitboard occ) {
    Bitboard attacks = 0;
    int r = rank_of(sq), f = file_of(sq);

    for (int rr = r+1, ff = f+1; rr <= 7 && ff <= 7; rr++, ff++) {
        Bitboard b = 1ULL << (rr * 8 + ff);
        attacks |= b;
        if (occ & b) break;
    }
    for (int rr = r+1, ff = f-1; rr <= 7 && ff >= 0; rr++, ff--) {
        Bitboard b = 1ULL << (rr * 8 + ff);
        attacks |= b;
        if (occ & b) break;
    }
    for (int rr = r-1, ff = f+1; rr >= 0 && ff <= 7; rr--, ff++) {
        Bitboard b = 1ULL << (rr * 8 + ff);
        attacks |= b;
        if (occ & b) break;
    }
    for (int rr = r-1, ff = f-1; rr >= 0 && ff >= 0; rr--, ff--) {
        Bitboard b = 1ULL << (rr * 8 + ff);
        attacks |= b;
        if (occ & b) break;
    }
    return attacks;
}

// Compute the relevant occupancy mask (excluding edges) for rook
static Bitboard rook_mask(Square sq) {
    Bitboard mask = 0;
    int r = rank_of(sq), f = file_of(sq);
    for (int rr = r+1; rr <= 6; rr++) mask |= 1ULL << (rr * 8 + f);
    for (int rr = r-1; rr >= 1; rr--) mask |= 1ULL << (rr * 8 + f);
    for (int ff = f+1; ff <= 6; ff++) mask |= 1ULL << (r * 8 + ff);
    for (int ff = f-1; ff >= 1; ff--) mask |= 1ULL << (r * 8 + ff);
    return mask;
}

// Compute the relevant occupancy mask (excluding edges) for bishop
static Bitboard bishop_mask(Square sq) {
    Bitboard mask = 0;
    int r = rank_of(sq), f = file_of(sq);
    for (int rr = r+1, ff = f+1; rr <= 6 && ff <= 6; rr++, ff++) mask |= 1ULL << (rr * 8 + ff);
    for (int rr = r+1, ff = f-1; rr <= 6 && ff >= 1; rr++, ff--) mask |= 1ULL << (rr * 8 + ff);
    for (int rr = r-1, ff = f+1; rr >= 1 && ff <= 6; rr--, ff++) mask |= 1ULL << (rr * 8 + ff);
    for (int rr = r-1, ff = f-1; rr >= 1 && ff >= 1; rr--, ff--) mask |= 1ULL << (rr * 8 + ff);
    return mask;
}

// Enumerate subsets of mask using Carry-Rippler
static void enumerate_subsets(Bitboard mask, Bitboard* subsets, int& count) {
    count = 0;
    Bitboard sub = 0;
    do {
        subsets[count++] = sub;
        sub = (sub - mask) & mask;
    } while (sub);
}

void magic_init() {
    // Initialize rook magics
    Bitboard* rook_ptr = RookTable;
    for (int sq = 0; sq < 64; sq++) {
        RookMagics[sq].mask = rook_mask(Square(sq));
        RookMagics[sq].magic = RookMagicNumbers[sq];
        RookMagics[sq].shift = RookShifts[sq];
        RookMagics[sq].attacks = rook_ptr;

        int bits = 64 - RookShifts[sq];
        int table_size = 1 << bits;
        rook_ptr += table_size;

        // Fill attack table
        Bitboard subsets[4096];
        int count;
        enumerate_subsets(RookMagics[sq].mask, subsets, count);

        for (int i = 0; i < count; i++) {
            unsigned idx = RookMagics[sq].index(subsets[i]);
            RookMagics[sq].attacks[idx] = compute_rook_attacks(Square(sq), subsets[i]);
        }
    }

    // Initialize bishop magics
    Bitboard* bishop_ptr = BishopTable;
    for (int sq = 0; sq < 64; sq++) {
        BishopMagics[sq].mask = bishop_mask(Square(sq));
        BishopMagics[sq].magic = BishopMagicNumbers[sq];
        BishopMagics[sq].shift = BishopShifts[sq];
        BishopMagics[sq].attacks = bishop_ptr;

        int bits = 64 - BishopShifts[sq];
        int table_size = 1 << bits;
        bishop_ptr += table_size;

        Bitboard subsets[4096];
        int count;
        enumerate_subsets(BishopMagics[sq].mask, subsets, count);

        for (int i = 0; i < count; i++) {
            unsigned idx = BishopMagics[sq].index(subsets[i]);
            BishopMagics[sq].attacks[idx] = compute_bishop_attacks(Square(sq), subsets[i]);
        }
    }
}

// Public lookup functions
Bitboard bishop_attacks(Square s, Bitboard occ) {
    return BishopMagics[s].attacks[BishopMagics[s].index(occ)];
}

Bitboard rook_attacks(Square s, Bitboard occ) {
    return RookMagics[s].attacks[RookMagics[s].index(occ)];
}

} // namespace alphaclaude
