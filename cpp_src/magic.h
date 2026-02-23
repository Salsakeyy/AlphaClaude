#pragma once

#include "types.h"

namespace alphaclaude {

// ============================================================
// Magic bitboard structures
// ============================================================

struct Magic {
    Bitboard* attacks;  // pointer into attack table
    Bitboard mask;      // relevant occupancy mask (excludes edges)
    Bitboard magic;     // magic number
    int shift;          // 64 - bits

    unsigned index(Bitboard occ) const {
        return unsigned(((occ & mask) * magic) >> shift);
    }
};

extern Magic BishopMagics[64];
extern Magic RookMagics[64];

// Total attack table storage
extern Bitboard BishopTable[0x1480]; // 5248
extern Bitboard RookTable[0x19000]; // 102400

void magic_init();

} // namespace alphaclaude
