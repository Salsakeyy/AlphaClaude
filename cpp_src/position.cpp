#include "position.h"
#include <sstream>
#include <cstring>
#include <random>
#include <algorithm>

namespace alphaclaude {

// ============================================================
// Zobrist
// ============================================================

namespace Zobrist {
    uint64_t psq[PIECE_NB][SQUARE_NB];
    uint64_t ep[FILE_NB];
    uint64_t castling[16];
    uint64_t side;

    void init() {
        std::mt19937_64 rng(0xDEADBEEF12345678ULL);
        for (int p = 0; p < PIECE_NB; p++)
            for (int s = 0; s < SQUARE_NB; s++)
                psq[p][s] = rng();
        for (int f = 0; f < FILE_NB; f++)
            ep[f] = rng();
        for (int i = 0; i < 16; i++)
            castling[i] = rng();
        side = rng();
    }
}

// ============================================================
// Castling helpers
// ============================================================

// Squares that must be empty for castling
static const Bitboard CastlePath[16] = {
    0, // 0: no castling
    (1ULL << SQ_F1) | (1ULL << SQ_G1),  // 1: WHITE_OO
    (1ULL << SQ_D1) | (1ULL << SQ_C1) | (1ULL << SQ_B1),  // 2: WHITE_OOO
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0
};

// The rook squares for each castling type
static const Square CastleRookFrom[16] = {
    SQ_NONE, SQ_H1, SQ_A1, SQ_NONE, SQ_H8, SQ_NONE, SQ_NONE, SQ_NONE,
    SQ_A8, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE
};
static const Square CastleRookTo[16] = {
    SQ_NONE, SQ_F1, SQ_D1, SQ_NONE, SQ_F8, SQ_NONE, SQ_NONE, SQ_NONE,
    SQ_D8, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE
};
static const Square CastleKingTo[16] = {
    SQ_NONE, SQ_G1, SQ_C1, SQ_NONE, SQ_G8, SQ_NONE, SQ_NONE, SQ_NONE,
    SQ_C8, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE
};

// Castling rights update table: after a piece moves from/to a square, AND with this
static CastlingRight castling_rights_mask[SQUARE_NB];

static void init_castling_masks() {
    for (int s = 0; s < 64; s++)
        castling_rights_mask[s] = ALL_CASTLING;

    castling_rights_mask[SQ_A1] = CastlingRight(ALL_CASTLING & ~WHITE_OOO);
    castling_rights_mask[SQ_E1] = CastlingRight(ALL_CASTLING & ~(WHITE_OO | WHITE_OOO));
    castling_rights_mask[SQ_H1] = CastlingRight(ALL_CASTLING & ~WHITE_OO);
    castling_rights_mask[SQ_A8] = CastlingRight(ALL_CASTLING & ~BLACK_OOO);
    castling_rights_mask[SQ_E8] = CastlingRight(ALL_CASTLING & ~(BLACK_OO | BLACK_OOO));
    castling_rights_mask[SQ_H8] = CastlingRight(ALL_CASTLING & ~BLACK_OO);
}

// ============================================================
// Position implementation
// ============================================================

Position::Position() {
    set_startpos();
}

Position::Position(const std::string& fen) {
    set(fen);
}

void Position::put_piece(Piece p, Square s) {
    board_[s] = p;
    Bitboard bb = square_bb(s);
    by_type_bb_[type_of(p)] |= bb;
    by_type_bb_[0] |= bb; // all pieces
    by_color_bb_[color_of(p)] |= bb;
    if (type_of(p) == KING)
        king_sq_[color_of(p)] = s;
}

void Position::remove_piece(Square s) {
    Piece p = board_[s];
    Bitboard bb = square_bb(s);
    by_type_bb_[type_of(p)] ^= bb;
    by_type_bb_[0] ^= bb;
    by_color_bb_[color_of(p)] ^= bb;
    board_[s] = NO_PIECE;
}

void Position::move_piece(Square from, Square to) {
    Piece p = board_[from];
    Bitboard from_to = square_bb(from) | square_bb(to);
    by_type_bb_[type_of(p)] ^= from_to;
    by_type_bb_[0] ^= from_to;
    by_color_bb_[color_of(p)] ^= from_to;
    board_[from] = NO_PIECE;
    board_[to] = p;
    if (type_of(p) == KING)
        king_sq_[color_of(p)] = to;
}

void Position::set_startpos() {
    set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Position::set(const std::string& fen) {
    static bool tables_initialized = false;
    if (!tables_initialized) {
        Zobrist::init();
        bitboard_init();
        init_castling_masks();
        tables_initialized = true;
    }

    std::memset(board_, 0, sizeof(board_));
    std::memset(by_type_bb_, 0, sizeof(by_type_bb_));
    std::memset(by_color_bb_, 0, sizeof(by_color_bb_));
    checkers_ = 0;
    blockers_[WHITE] = blockers_[BLACK] = 0;
    pinners_[WHITE] = pinners_[BLACK] = 0;

    state_stack_.clear();
    st_ = &root_si_;
    std::memset(st_, 0, sizeof(StateInfo));
    st_->ep_square = SQ_NONE;

    std::istringstream ss(fen);
    std::string piece_str, color_str, castling_str, ep_str;
    int halfmove = 0, fullmove = 1;

    ss >> piece_str >> color_str >> castling_str >> ep_str >> halfmove >> fullmove;

    // Parse pieces
    int sq = 56; // start from A8
    for (char c : piece_str) {
        if (c == '/') {
            sq -= 16; // go to start of next rank down
        } else if (c >= '1' && c <= '8') {
            sq += (c - '0');
        } else {
            Piece p = NO_PIECE;
            switch (c) {
                case 'P': p = W_PAWN; break;   case 'N': p = W_KNIGHT; break;
                case 'B': p = W_BISHOP; break;  case 'R': p = W_ROOK; break;
                case 'Q': p = W_QUEEN; break;   case 'K': p = W_KING; break;
                case 'p': p = B_PAWN; break;   case 'n': p = B_KNIGHT; break;
                case 'b': p = B_BISHOP; break;  case 'r': p = B_ROOK; break;
                case 'q': p = B_QUEEN; break;   case 'k': p = B_KING; break;
                default: break;
            }
            if (p != NO_PIECE) {
                put_piece(p, Square(sq));
                sq++;
            }
        }
    }

    side_to_move_ = (color_str == "w") ? WHITE : BLACK;
    fullmove_ = fullmove;

    // Parse castling
    st_->castling = NO_CASTLING;
    if (castling_str != "-") {
        for (char c : castling_str) {
            switch (c) {
                case 'K': st_->castling |= WHITE_OO; break;
                case 'Q': st_->castling |= WHITE_OOO; break;
                case 'k': st_->castling |= BLACK_OO; break;
                case 'q': st_->castling |= BLACK_OOO; break;
                default: break;
            }
        }
    }

    // Parse EP
    if (ep_str != "-" && ep_str.length() == 2) {
        int f = ep_str[0] - 'a';
        int r = ep_str[1] - '1';
        st_->ep_square = make_square(File(f), Rank(r));
    }

    st_->halfmove_clock = halfmove;

    // Compute hash
    st_->hash = 0;
    for (int s = 0; s < 64; s++) {
        if (board_[s] != NO_PIECE)
            st_->hash ^= Zobrist::psq[board_[s]][s];
    }
    if (side_to_move_ == BLACK) st_->hash ^= Zobrist::side;
    st_->hash ^= Zobrist::castling[st_->castling];
    if (st_->ep_square != SQ_NONE)
        st_->hash ^= Zobrist::ep[file_of(st_->ep_square)];

    st_->repetition = 0;

    compute_check_info();
}

std::string Position::fen() const {
    std::string s;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            Piece p = board_[r * 8 + f];
            if (p == NO_PIECE) {
                empty++;
            } else {
                if (empty > 0) { s += char('0' + empty); empty = 0; }
                const char* piece_chars = " PNBRQK  pnbrqk";
                s += piece_chars[p];
            }
        }
        if (empty > 0) s += char('0' + empty);
        if (r > 0) s += '/';
    }
    s += ' ';
    s += (side_to_move_ == WHITE) ? 'w' : 'b';
    s += ' ';

    std::string castle_str;
    if (st_->castling & WHITE_OO) castle_str += 'K';
    if (st_->castling & WHITE_OOO) castle_str += 'Q';
    if (st_->castling & BLACK_OO) castle_str += 'k';
    if (st_->castling & BLACK_OOO) castle_str += 'q';
    s += castle_str.empty() ? "-" : castle_str;

    s += ' ';
    if (st_->ep_square != SQ_NONE) {
        s += char('a' + file_of(st_->ep_square));
        s += char('1' + rank_of(st_->ep_square));
    } else {
        s += '-';
    }

    s += ' ' + std::to_string(st_->halfmove_clock);
    s += ' ' + std::to_string(fullmove_);
    return s;
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (PawnAttacks[BLACK][s] & pieces(WHITE, PAWN))
         | (PawnAttacks[WHITE][s] & pieces(BLACK, PAWN))
         | (KnightAttacks[s]     & pieces(KNIGHT))
         | (bishop_attacks(s, occ) & pieces(BISHOP, QUEEN))
         | (rook_attacks(s, occ)   & pieces(ROOK, QUEEN))
         | (KingAttacks[s]       & pieces(KING));
}

Bitboard Position::slider_blockers(Bitboard sliders, Square s, Bitboard& pinners_out) const {
    Bitboard blockers = 0;
    pinners_out = 0;

    // Snipers: sliders that could attack s if blockers were removed
    Bitboard snipers = ((PseudoRookAttacks[s] & pieces(ROOK, QUEEN))
                      | (PseudoBishopAttacks[s] & pieces(BISHOP, QUEEN))) & sliders;

    Bitboard occ = pieces() ^ snipers;

    while (snipers) {
        Square sniper = pop_lsb(snipers);
        Bitboard between = BetweenBB[s][sniper] & occ;

        if (between && !more_than_one(between)) {
            blockers |= between;
            if (between & pieces(color_of(piece_on(s))))
                pinners_out |= square_bb(sniper);
        }
    }
    return blockers;
}

void Position::compute_check_info() {
    Color us = side_to_move_;
    Color them = ~us;
    Square ksq = king_sq_[us];

    checkers_ = attackers_to(ksq) & pieces(them);
    blockers_[WHITE] = slider_blockers(pieces(BLACK), king_sq_[WHITE], pinners_[BLACK]);
    blockers_[BLACK] = slider_blockers(pieces(WHITE), king_sq_[BLACK], pinners_[WHITE]);
}

void Position::do_move(Move m, StateInfo& new_si) {
    // Save old state
    std::memcpy(&new_si, st_, sizeof(StateInfo));
    state_stack_.push_back(st_);
    st_ = &new_si;

    Color us = side_to_move_;
    Color them = ~us;
    Square from = m.from();
    Square to = m.to();
    Piece moved = board_[from];
    Piece captured = (m.flag() == MOVE_EN_PASSANT) ? make_piece(them, PAWN) : board_[to];

    st_->captured = captured;

    // Update hash for side change
    st_->hash ^= Zobrist::side;

    // Remove old EP hash
    if (st_->ep_square != SQ_NONE)
        st_->hash ^= Zobrist::ep[file_of(st_->ep_square)];

    // Remove old castling hash
    st_->hash ^= Zobrist::castling[st_->castling];

    // Halfmove clock
    st_->halfmove_clock++;
    if (type_of(moved) == PAWN || captured != NO_PIECE)
        st_->halfmove_clock = 0;

    // Reset EP
    st_->ep_square = SQ_NONE;

    if (m.flag() == MOVE_CASTLING) {
        // Determine which castling
        CastlingRight cr = (to > from) ? king_side(us) : queen_side(us);
        Square rook_from = CastleRookFrom[cr];
        Square rook_to = CastleRookTo[cr];
        Square king_to = CastleKingTo[cr];

        // Remove king and rook from old squares
        st_->hash ^= Zobrist::psq[moved][from];
        st_->hash ^= Zobrist::psq[make_piece(us, ROOK)][rook_from];

        remove_piece(from);
        remove_piece(rook_from);
        // NOTE: board_[to] might be SQ_NONE or the rook; clear it safely
        board_[to] = NO_PIECE; // in case to == rook_from

        put_piece(moved, king_to);
        put_piece(make_piece(us, ROOK), rook_to);

        st_->hash ^= Zobrist::psq[moved][king_to];
        st_->hash ^= Zobrist::psq[make_piece(us, ROOK)][rook_to];
    } else {
        // Handle captures
        if (captured != NO_PIECE) {
            Square cap_sq = to;
            if (m.flag() == MOVE_EN_PASSANT) {
                cap_sq = to + (us == WHITE ? SOUTH : NORTH);
            }
            st_->hash ^= Zobrist::psq[captured][cap_sq];
            remove_piece(cap_sq);
        }

        // Move the piece
        st_->hash ^= Zobrist::psq[moved][from];

        if (m.flag() == MOVE_PROMOTION) {
            remove_piece(from);
            Piece promo = make_piece(us, m.promotion_type());
            put_piece(promo, to);
            st_->hash ^= Zobrist::psq[promo][to];
        } else {
            move_piece(from, to);
            st_->hash ^= Zobrist::psq[moved][to];
        }

        // Set EP square for double pawn push
        if (type_of(moved) == PAWN && std::abs(int(to) - int(from)) == 16) {
            Square ep = Square((int(from) + int(to)) / 2);
            // Only set EP if opponent pawn can actually capture
            if (PawnAttacks[us][ep] & pieces(them, PAWN)) {
                st_->ep_square = ep;
                st_->hash ^= Zobrist::ep[file_of(ep)];
            }
        }
    }

    // Update castling rights
    st_->castling &= castling_rights_mask[from];
    st_->castling &= castling_rights_mask[to];
    st_->hash ^= Zobrist::castling[st_->castling];

    // Switch side
    side_to_move_ = them;
    if (us == BLACK) fullmove_++;

    // Update repetition
    st_->repetition = 0;
    int end = std::min(int(state_stack_.size()), st_->halfmove_clock);
    if (end >= 4) {
        // Walk back through state history
        int cnt = state_stack_.size();
        for (int i = 4; i <= end; i += 2) {
            int idx = cnt - i;
            if (idx < 0) break;
            StateInfo* prev = state_stack_[idx];
            if (prev->hash == st_->hash) {
                st_->repetition = prev->repetition ? 2 : 1;
                break;
            }
        }
    }

    compute_check_info();
}

void Position::undo_move(Move m) {
    Color them = side_to_move_; // current side (was "them" when move was made)
    Color us = ~them;
    Square from = m.from();
    Square to = m.to();

    if (m.flag() == MOVE_CASTLING) {
        CastlingRight cr = (to > from) ? king_side(us) : queen_side(us);
        Square rook_from = CastleRookFrom[cr];
        Square rook_to = CastleRookTo[cr];
        Square king_to = CastleKingTo[cr];

        remove_piece(king_to);
        remove_piece(rook_to);
        board_[king_to] = NO_PIECE;
        board_[rook_to] = NO_PIECE;
        put_piece(make_piece(us, KING), from);
        put_piece(make_piece(us, ROOK), rook_from);
    } else {
        if (m.flag() == MOVE_PROMOTION) {
            remove_piece(to);
            put_piece(make_piece(us, PAWN), from);
        } else {
            move_piece(to, from);
        }

        // Restore captured piece
        if (st_->captured != NO_PIECE) {
            Square cap_sq = to;
            if (m.flag() == MOVE_EN_PASSANT)
                cap_sq = to + (us == WHITE ? SOUTH : NORTH);
            put_piece(st_->captured, cap_sq);
        }
    }

    // Restore side
    side_to_move_ = us;
    if (us == BLACK) fullmove_--;

    // Restore state
    st_ = state_stack_.back();
    state_stack_.pop_back();

    compute_check_info();
}

bool Position::gives_check(Move m) const {
    Square from = m.from();
    Square to = m.to();
    Color us = side_to_move_;
    Square their_king = king_sq_[~us];

    PieceType pt = type_of(piece_on(from));

    // Direct check from destination square
    if (m.flag() == MOVE_PROMOTION)
        pt = m.promotion_type();

    if (m.flag() != MOVE_CASTLING) {
        Bitboard occ = (pieces() ^ square_bb(from)) | square_bb(to);
        if (m.flag() == MOVE_EN_PASSANT)
            occ ^= square_bb(to + (us == WHITE ? SOUTH : NORTH));

        switch (pt) {
            case PAWN:   if (PawnAttacks[us][to] & square_bb(their_king)) return true; break;
            case KNIGHT: if (KnightAttacks[to] & square_bb(their_king)) return true; break;
            case BISHOP: if (bishop_attacks(to, occ) & square_bb(their_king)) return true; break;
            case ROOK:   if (rook_attacks(to, occ) & square_bb(their_king)) return true; break;
            case QUEEN:  if (queen_attacks(to, occ) & square_bb(their_king)) return true; break;
            default: break;
        }

        // Discovered check
        if (blockers_for_king(~us) & square_bb(from)) {
            if (!aligned(from, to, their_king))
                return true;
        }
    }

    return false;
}

bool Position::is_legal(Move m) const {
    Color us = side_to_move_;
    Square from = m.from();
    Square to = m.to();
    Square ksq = king_sq_[us];

    if (m.flag() == MOVE_EN_PASSANT) {
        Square cap_sq = to + (us == WHITE ? SOUTH : NORTH);
        Bitboard occ = (pieces() ^ square_bb(from) ^ square_bb(cap_sq)) | square_bb(to);
        return !(rook_attacks(ksq, occ) & pieces(~us, ROOK, QUEEN))
            && !(bishop_attacks(ksq, occ) & pieces(~us, BISHOP, QUEEN));
    }

    if (m.flag() == MOVE_CASTLING) {
        // Check that none of the king's path squares are attacked
        Square king_to = CastleKingTo[to > from ? king_side(us) : queen_side(us)];
        int step = (king_to > ksq) ? 1 : -1;
        for (Square s = ksq + step; ; s = s + step) {
            if (is_attacked_by(~us, s)) return false;
            if (s == king_to) break;
        }
        // Also king itself can't be in check (handled by caller usually)
        return !checkers_;
    }

    // King move: check that destination is not attacked
    if (from == ksq) {
        Bitboard occ = pieces() ^ square_bb(from);
        return !(attackers_to(to, occ) & pieces(~us));
    }

    // If piece is pinned, it can only move along the pin line
    if (blockers_for_king(us) & square_bb(from)) {
        return aligned(from, to, ksq);
    }

    return true;
}

Move Position::parse_uci(const std::string& uci) const {
    Square from = make_square(File(uci[0] - 'a'), Rank(uci[1] - '1'));
    Square to = make_square(File(uci[2] - 'a'), Rank(uci[3] - '1'));

    if (uci.length() == 5) {
        PieceType promo = QUEEN;
        switch (uci[4]) {
            case 'n': promo = KNIGHT; break;
            case 'b': promo = BISHOP; break;
            case 'r': promo = ROOK; break;
            case 'q': promo = QUEEN; break;
        }
        return Move::make(from, to, MOVE_PROMOTION, promo);
    }

    // Detect castling (king moving 2+ squares)
    if (type_of(piece_on(from)) == KING && std::abs(file_of(from) - file_of(to)) >= 2) {
        return Move::make(from, to, MOVE_CASTLING);
    }

    // Detect en passant
    if (type_of(piece_on(from)) == PAWN && to == ep_square()) {
        return Move::make(from, to, MOVE_EN_PASSANT);
    }

    return Move::make(from, to);
}

} // namespace alphaclaude
