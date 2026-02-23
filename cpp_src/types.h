#pragma once

#include <cstdint>
#include <string>
#include <cassert>

namespace alphaclaude {

// ============================================================
// Fundamental types
// ============================================================

using Bitboard = uint64_t;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    NO_PIECE_TYPE = 0,
    PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6,
    PIECE_TYPE_NB = 7
};

enum Piece : int {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT = 2, W_BISHOP = 3, W_ROOK = 4, W_QUEEN = 5, W_KING = 6,
    B_PAWN = 9, B_KNIGHT = 10, B_BISHOP = 11, B_ROOK = 12, B_QUEEN = 13, B_KING = 14,
    PIECE_NB = 16
};

constexpr Piece make_piece(Color c, PieceType pt) {
    return Piece((c << 3) | pt);
}

constexpr Color color_of(Piece p) {
    assert(p != NO_PIECE);
    return Color(p >> 3);
}

constexpr PieceType type_of(Piece p) {
    return PieceType(p & 7);
}

// ============================================================
// Squares: A1=0, B1=1, ..., H8=63
// ============================================================

enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE = 64,
    SQUARE_NB = 64
};

enum File : int { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB };
enum Rank : int { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB };

constexpr Square make_square(File f, Rank r) { return Square((r << 3) | f); }
constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }

constexpr Rank relative_rank(Color c, Rank r) { return Rank(r ^ (c * 7)); }
constexpr Rank relative_rank(Color c, Square s) { return relative_rank(c, rank_of(s)); }
constexpr Square relative_square(Color c, Square s) { return Square(s ^ (c * 56)); }

// Flip square vertically (for black perspective)
constexpr Square flip_sq(Square s) { return Square(s ^ 56); }

constexpr Square operator+(Square s, int d) { return Square(int(s) + d); }
constexpr Square operator-(Square s, int d) { return Square(int(s) - d); }
inline Square& operator+=(Square& s, int d) { return s = s + d; }
inline Square& operator-=(Square& s, int d) { return s = s - d; }
inline Square& operator++(Square& s) { return s = Square(int(s) + 1); }

// ============================================================
// Directions
// ============================================================

enum Direction : int {
    NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
    NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9,
    NORTH_NORTH = 16, SOUTH_SOUTH = -16
};

constexpr Direction pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }

// ============================================================
// Move: 16-bit packed
// Bits 0-5: from square
// Bits 6-11: to square
// Bits 12-13: flags (0=normal, 1=promotion, 2=en passant, 3=castling)
// Bits 14-15: promotion piece type (0=knight, 1=bishop, 2=rook, 3=queen)
// ============================================================

enum MoveFlag : int {
    MOVE_NORMAL     = 0,
    MOVE_PROMOTION  = 1,
    MOVE_EN_PASSANT = 2,
    MOVE_CASTLING   = 3
};

struct Move {
    uint16_t data;

    constexpr Move() : data(0) {}
    explicit constexpr Move(uint16_t d) : data(d) {}

    static Move make(Square from, Square to, MoveFlag flag = MOVE_NORMAL, PieceType promo = KNIGHT) {
        return Move(uint16_t(from | (to << 6) | (flag << 12) | ((promo - KNIGHT) << 14)));
    }

    Square from() const { return Square(data & 0x3F); }
    Square to() const { return Square((data >> 6) & 0x3F); }
    MoveFlag flag() const { return MoveFlag((data >> 12) & 0x3); }
    PieceType promotion_type() const { return PieceType(((data >> 14) & 0x3) + KNIGHT); }

    bool operator==(Move other) const { return data == other.data; }
    bool operator!=(Move other) const { return data != other.data; }
    explicit operator bool() const { return data != 0; }

    std::string to_uci() const {
        std::string s;
        s += char('a' + file_of(from()));
        s += char('1' + rank_of(from()));
        s += char('a' + file_of(to()));
        s += char('1' + rank_of(to()));
        if (flag() == MOVE_PROMOTION) {
            const char promo_chars[] = "nbrq";
            s += promo_chars[promotion_type() - KNIGHT];
        }
        return s;
    }
};

constexpr Move MOVE_NONE = Move();

// ============================================================
// Castling rights (stored as bitfield)
// ============================================================

enum CastlingRight : int {
    NO_CASTLING = 0,
    WHITE_OO  = 1,
    WHITE_OOO = 2,
    BLACK_OO  = 4,
    BLACK_OOO = 8,
    ALL_CASTLING = 15
};

constexpr CastlingRight operator|(CastlingRight a, CastlingRight b) {
    return CastlingRight(int(a) | int(b));
}
constexpr CastlingRight operator&(CastlingRight a, CastlingRight b) {
    return CastlingRight(int(a) & int(b));
}
constexpr CastlingRight operator~(CastlingRight a) {
    return CastlingRight(~int(a) & 15);
}
inline CastlingRight& operator|=(CastlingRight& a, CastlingRight b) { return a = a | b; }
inline CastlingRight& operator&=(CastlingRight& a, CastlingRight b) { return a = a & b; }

constexpr CastlingRight king_side(Color c) { return c == WHITE ? WHITE_OO : BLACK_OO; }
constexpr CastlingRight queen_side(Color c) { return c == WHITE ? WHITE_OOO : BLACK_OOO; }

// ============================================================
// MoveList: stack-allocated list of moves
// ============================================================

struct MoveList {
    Move moves[256];
    int count = 0;

    void add(Move m) { moves[count++] = m; }
    Move operator[](int i) const { return moves[i]; }
    int size() const { return count; }
    Move* begin() { return moves; }
    Move* end() { return moves + count; }
    const Move* begin() const { return moves; }
    const Move* end() const { return moves + count; }
};

// ============================================================
// State info for unmake
// ============================================================

struct StateInfo {
    CastlingRight castling;
    Square ep_square;
    int halfmove_clock;
    Piece captured;
    uint64_t hash;
    uint64_t pawn_hash;
    int repetition;  // 0 = none, 1 = seen once before, 2 = threefold
};

} // namespace alphaclaude
