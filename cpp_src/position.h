#pragma once

#include "types.h"
#include "bitboard.h"
#include <string>
#include <vector>

namespace alphaclaude {

// ============================================================
// Zobrist keys
// ============================================================

namespace Zobrist {
    extern uint64_t psq[PIECE_NB][SQUARE_NB];
    extern uint64_t ep[FILE_NB];
    extern uint64_t castling[16];
    extern uint64_t side;
    void init();
}

// ============================================================
// Position class
// ============================================================

class Position {
public:
    Position();
    Position(const std::string& fen);

    // Set up
    void set(const std::string& fen);
    void set_startpos();
    std::string fen() const;

    // Accessors
    Color side_to_move() const { return side_to_move_; }
    Piece piece_on(Square s) const { return board_[s]; }
    Bitboard pieces() const { return by_type_bb_[0]; } // all pieces (computed)
    Bitboard pieces(Color c) const { return by_color_bb_[c]; }
    Bitboard pieces(PieceType pt) const { return by_type_bb_[pt]; }
    Bitboard pieces(Color c, PieceType pt) const { return by_color_bb_[c] & by_type_bb_[pt]; }
    Bitboard pieces(PieceType pt1, PieceType pt2) const { return by_type_bb_[pt1] | by_type_bb_[pt2]; }
    Bitboard pieces(Color c, PieceType pt1, PieceType pt2) const { return by_color_bb_[c] & (by_type_bb_[pt1] | by_type_bb_[pt2]); }
    Square king_sq(Color c) const { return king_sq_[c]; }
    CastlingRight castling() const { return st_->castling; }
    Square ep_square() const { return st_->ep_square; }
    int halfmove_clock() const { return st_->halfmove_clock; }
    int fullmove_number() const { return fullmove_; }
    uint64_t hash() const { return st_->hash; }
    int repetition() const { return st_->repetition; }

    // Attackers
    Bitboard attackers_to(Square s, Bitboard occ) const;
    Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
    bool is_attacked_by(Color c, Square s) const { return attackers_to(s) & pieces(c); }
    Bitboard checkers() const { return checkers_; }
    Bitboard blockers_for_king(Color c) const { return blockers_[c]; }
    Bitboard pinners(Color c) const { return pinners_[c]; }

    // Move execution
    void do_move(Move m, StateInfo& new_si);
    void undo_move(Move m);

    // Helpers
    bool gives_check(Move m) const;
    bool is_legal(Move m) const; // assumes pseudo-legal
    Bitboard slider_blockers(Bitboard sliders, Square s, Bitboard& pinners_out) const;

    // Move from UCI string
    Move parse_uci(const std::string& uci) const;

    // Fix st_ after move construction/assignment.
    // After a move, st_ may point to the old object's root_si_.
    // Call this with the OLD (moved-from) object to redirect st_ if needed.
    void fix_st_after_move(const Position& old_pos) {
        if (st_ == &old_pos.root_si_) {
            st_ = &root_si_;
        }
    }

private:
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    void compute_check_info();

    Piece board_[SQUARE_NB];
    Bitboard by_type_bb_[PIECE_TYPE_NB];
    Bitboard by_color_bb_[COLOR_NB];
    Square king_sq_[COLOR_NB];

    Color side_to_move_;
    int fullmove_;

    // Check / pin info (recomputed after each move)
    Bitboard checkers_;
    Bitboard blockers_[COLOR_NB];
    Bitboard pinners_[COLOR_NB];

    // State history stack
    StateInfo root_si_;
    StateInfo* st_;
    std::vector<StateInfo*> state_stack_; // for undo tracking
};

} // namespace alphaclaude
