#include "game.h"
#include <cmath>
#include <algorithm>

namespace alphaclaude {

// ============================================================
// Move encoding helpers
// ============================================================

// Queen-move directions (from white's perspective): N, NE, E, SE, S, SW, W, NW
static const int QUEEN_DIR_DR[8] = { 1,  1,  0, -1, -1, -1,  0,  1};
static const int QUEEN_DIR_DF[8] = { 0,  1,  1,  1,  0, -1, -1, -1};

// Knight deltas (rank_delta, file_delta)
static const int KNIGHT_DR[8] = { 2,  2,  1,  1, -1, -1, -2, -2};
static const int KNIGHT_DF[8] = { 1, -1,  2, -2,  2, -2,  1, -1};

// Underpromotion directions (from white's perspective): left-capture, push, right-capture
// Encoded as file deltas: -1, 0, +1
static const int UNDERPROMO_DF[3] = {-1, 0, 1};
// Underpromotion piece types: knight=0, bishop=1, rook=2
static const PieceType UNDERPROMO_PT[3] = {KNIGHT, BISHOP, ROOK};

int GameState::encode_move(Move m, Color perspective) {
    Square from = m.from();
    Square to = m.to();

    // Flip for black's perspective
    if (perspective == BLACK) {
        from = flip_sq(from);
        to = flip_sq(to);
    }

    int from_rank = rank_of(from);
    int from_file = file_of(from);
    int to_rank = rank_of(to);
    int to_file = file_of(to);

    int dr = to_rank - from_rank;
    int df = to_file - from_file;

    int plane = -1;

    // Check for underpromotions first
    if (m.flag() == MOVE_PROMOTION && m.promotion_type() != QUEEN) {
        PieceType pt = m.promotion_type();
        int piece_idx = -1;
        for (int i = 0; i < 3; i++) {
            if (UNDERPROMO_PT[i] == pt) { piece_idx = i; break; }
        }
        int dir_idx = -1;
        for (int i = 0; i < 3; i++) {
            if (UNDERPROMO_DF[i] == df) { dir_idx = i; break; }
        }
        if (piece_idx >= 0 && dir_idx >= 0) {
            plane = 64 + piece_idx * 3 + dir_idx;
        }
    }
    // Knight moves
    else if (std::abs(dr) + std::abs(df) == 3 && dr != 0 && df != 0) {
        for (int i = 0; i < 8; i++) {
            if (KNIGHT_DR[i] == dr && KNIGHT_DF[i] == df) {
                plane = 56 + i;
                break;
            }
        }
    }
    // Queen-type moves (including queen promotions, rook moves, bishop moves, pawn pushes)
    else {
        int dist = std::max(std::abs(dr), std::abs(df));
        if (dist > 0) {
            int norm_dr = (dr == 0) ? 0 : dr / std::abs(dr);
            int norm_df = (df == 0) ? 0 : df / std::abs(df);
            for (int d = 0; d < 8; d++) {
                if (QUEEN_DIR_DR[d] == norm_dr && QUEEN_DIR_DF[d] == norm_df) {
                    plane = d * 7 + (dist - 1);
                    break;
                }
            }
        }
    }

    if (plane < 0) return -1; // should not happen for valid moves

    return from_rank * 8 * 73 + from_file * 73 + plane;
}

Move GameState::decode_move(int idx, Color perspective) {
    int plane = idx % 73;
    int from_file = (idx / 73) % 8;
    int from_rank = idx / (73 * 8);

    Square from = make_square(File(from_file), Rank(from_rank));
    Square to;
    MoveFlag flag = MOVE_NORMAL;
    PieceType promo = KNIGHT;

    if (plane < 56) {
        // Queen-type move
        int dir = plane / 7;
        int dist = (plane % 7) + 1;
        int to_rank = from_rank + QUEEN_DIR_DR[dir] * dist;
        int to_file = from_file + QUEEN_DIR_DF[dir] * dist;
        to = make_square(File(to_file), Rank(to_rank));

        // Check if this is a pawn reaching promotion rank
        if (from_rank == 6 && to_rank == 7) {
            flag = MOVE_PROMOTION;
            promo = QUEEN;
        }
    } else if (plane < 64) {
        // Knight move
        int knight_idx = plane - 56;
        int to_rank = from_rank + KNIGHT_DR[knight_idx];
        int to_file = from_file + KNIGHT_DF[knight_idx];
        to = make_square(File(to_file), Rank(to_rank));
    } else {
        // Underpromotion
        int underpromo_idx = plane - 64;
        int piece_idx = underpromo_idx / 3;
        int dir_idx = underpromo_idx % 3;
        int to_rank = from_rank + 1; // always one step forward (white perspective)
        int to_file = from_file + UNDERPROMO_DF[dir_idx];
        to = make_square(File(to_file), Rank(to_rank));
        flag = MOVE_PROMOTION;
        promo = UNDERPROMO_PT[piece_idx];
    }

    // Flip back for black's perspective
    if (perspective == BLACK) {
        from = flip_sq(from);
        to = flip_sq(to);
    }

    return Move::make(from, to, flag, promo);
}

// ============================================================
// GameState implementation
// ============================================================

static GameState::BoardSnapshot make_snapshot(const Position& pos) {
    GameState::BoardSnapshot snap;
    for (int pt = 0; pt < PIECE_TYPE_NB; pt++)
        snap.by_type[pt] = pos.pieces(PieceType(pt));
    snap.by_color[WHITE] = pos.pieces(WHITE);
    snap.by_color[BLACK] = pos.pieces(BLACK);
    snap.repetition = pos.repetition();
    return snap;
}

GameState::GameState() {
    reset();
}

GameState::GameState(const std::string& fen) {
    fen_history_.clear();
    move_history_.clear();
    fen_history_.push_back(fen);
    pos_.set(fen);
    state_history_.clear();
    board_history_.clear();
    board_history_.push_back(make_snapshot(pos_));
    move_count_ = 0;
}

GameState::GameState(const GameState& other)
    : board_history_(other.board_history_),
      fen_history_(other.fen_history_),
      move_history_(other.move_history_),
      move_count_(other.move_count_)
{
    // Replay moves from initial FEN to get a valid Position with correct st_ pointer
    if (fen_history_.empty()) {
        pos_.set_startpos();
    } else {
        pos_.set(fen_history_[0]);
    }
    state_history_.clear();
    for (const auto& uci : move_history_) {
        Move m = pos_.parse_uci(uci);
        state_history_.emplace_back();
        pos_.do_move(m, state_history_.back());
    }
    // board_history_ is already copied (safe plain-data struct)
}

GameState& GameState::operator=(const GameState& other) {
    if (this != &other) {
        GameState tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

GameState::GameState(GameState&& other) noexcept
    : pos_(std::move(other.pos_)),
      state_history_(std::move(other.state_history_)),
      board_history_(std::move(other.board_history_)),
      fen_history_(std::move(other.fen_history_)),
      move_history_(std::move(other.move_history_)),
      move_count_(other.move_count_)
{
    // Fix st_ pointer: if it was pointing at other's root_si_, redirect to ours
    pos_.fix_st_after_move(other.pos_);
}

GameState& GameState::operator=(GameState&& other) noexcept {
    if (this != &other) {
        pos_ = std::move(other.pos_);
        state_history_ = std::move(other.state_history_);
        board_history_ = std::move(other.board_history_);
        fen_history_ = std::move(other.fen_history_);
        move_history_ = std::move(other.move_history_);
        move_count_ = other.move_count_;
        // Fix st_ pointer: if it was pointing at other's root_si_, redirect to ours
        pos_.fix_st_after_move(other.pos_);
    }
    return *this;
}

void GameState::reset() {
    pos_.set_startpos();
    state_history_.clear();
    board_history_.clear();
    fen_history_.clear();
    move_history_.clear();
    fen_history_.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    board_history_.push_back(make_snapshot(pos_));
    move_count_ = 0;
}

void GameState::make_move(Move m) {
    move_history_.push_back(m.to_uci());
    state_history_.emplace_back();
    pos_.do_move(m, state_history_.back());
    board_history_.push_back(make_snapshot(pos_));
    move_count_++;
}

void GameState::make_move_uci(const std::string& uci) {
    Move m = pos_.parse_uci(uci);
    make_move(m);
}

MoveList GameState::legal_moves() const {
    MoveList list;
    generate_legal(pos_, list);
    return list;
}

bool GameState::is_draw() const {
    // 50-move rule
    if (pos_.halfmove_clock() >= 100) return true;

    // Threefold repetition
    if (pos_.repetition() >= 2) return true;

    // Insufficient material
    Bitboard all = pos_.pieces();
    int piece_count = popcount(all);
    if (piece_count == 2) return true; // K vs K
    if (piece_count == 3) {
        // K+B vs K or K+N vs K
        if (pos_.pieces(BISHOP) || pos_.pieces(KNIGHT)) return true;
    }
    if (piece_count == 4) {
        // K+B vs K+B (same color bishops)
        Bitboard bishops = pos_.pieces(BISHOP);
        if (popcount(bishops) == 2) {
            Square b1 = lsb(bishops);
            Bitboard bishops2 = bishops & (bishops - 1);
            Square b2 = lsb(bishops2);
            // Same color square check
            if (((rank_of(b1) + file_of(b1)) % 2) == ((rank_of(b2) + file_of(b2)) % 2))
                return true;
        }
    }

    // Stalemate
    if (!pos_.checkers()) {
        MoveList moves;
        generate_legal(pos_, moves);
        if (moves.size() == 0) return true;
    }

    return false;
}

bool GameState::is_terminal() const {
    MoveList moves;
    generate_legal(pos_, moves);
    if (moves.size() == 0) return true; // checkmate or stalemate

    // Draw conditions
    if (pos_.halfmove_clock() >= 100) return true;
    if (pos_.repetition() >= 2) return true;

    // Insufficient material
    Bitboard all = pos_.pieces();
    int piece_count = popcount(all);
    if (piece_count == 2) return true;
    if (piece_count == 3 && (pos_.pieces(BISHOP) || pos_.pieces(KNIGHT))) return true;
    if (piece_count == 4) {
        Bitboard bishops = pos_.pieces(BISHOP);
        if (popcount(bishops) == 2) {
            Square b1 = lsb(bishops);
            Bitboard bishops2 = bishops & (bishops - 1);
            Square b2 = lsb(bishops2);
            if (((rank_of(b1) + file_of(b1)) % 2) == ((rank_of(b2) + file_of(b2)) % 2))
                return true;
        }
    }

    return false;
}

float GameState::terminal_value() const {
    // Must only be called when is_terminal() is true
    MoveList moves;
    generate_legal(pos_, moves);

    if (moves.size() == 0 && pos_.checkers()) {
        // Checkmate: current player lost
        return -1.0f;
    }

    // All other terminal states are draws
    return 0.0f;
}

void GameState::encode_board(const BoardSnapshot& snap, Color perspective, float* planes) const {
    // 14 planes per step:
    // 0-5: own pieces (P,N,B,R,Q,K)
    // 6-11: opponent pieces (P,N,B,R,Q,K)
    // 12: repetition count >= 1
    // 13: repetition count >= 2

    Color own = perspective;
    Color opp = ~perspective;

    for (int pt = PAWN; pt <= KING; pt++) {
        Bitboard bb = snap.by_type[pt] & snap.by_color[own];
        int plane_idx = pt - 1;
        while (bb) {
            Square sq = pop_lsb(bb);
            Square mapped = (perspective == BLACK) ? flip_sq(sq) : sq;
            int r = rank_of(mapped), f = file_of(mapped);
            planes[plane_idx * 64 + r * 8 + f] = 1.0f;
        }
    }

    for (int pt = PAWN; pt <= KING; pt++) {
        Bitboard bb = snap.by_type[pt] & snap.by_color[opp];
        int plane_idx = 6 + pt - 1;
        while (bb) {
            Square sq = pop_lsb(bb);
            Square mapped = (perspective == BLACK) ? flip_sq(sq) : sq;
            int r = rank_of(mapped), f = file_of(mapped);
            planes[plane_idx * 64 + r * 8 + f] = 1.0f;
        }
    }

    // Repetition planes
    int rep = snap.repetition;
    if (rep >= 1) {
        for (int i = 0; i < 64; i++) planes[12 * 64 + i] = 1.0f;
    }
    if (rep >= 2) {
        for (int i = 0; i < 64; i++) planes[13 * 64 + i] = 1.0f;
    }
}

void GameState::get_nn_input(float* output) const {
    std::memset(output, 0, TOTAL_PLANES * 64 * sizeof(float));

    Color perspective = pos_.side_to_move();

    // Fill history steps (most recent first)
    int hist_size = (int)board_history_.size();
    for (int step = 0; step < NUM_HISTORY_STEPS; step++) {
        int hist_idx = hist_size - 1 - step;
        if (hist_idx < 0) break; // no more history, leave as zeros

        float* planes = output + step * PLANES_PER_STEP * 64;
        encode_board(board_history_[hist_idx], perspective, planes);
    }

    // Auxiliary planes (constant across all squares)
    float* aux = output + NUM_HISTORY_STEPS * PLANES_PER_STEP * 64;

    // Plane 0: color (1 if black to move, 0 if white)
    float color_val = (perspective == BLACK) ? 1.0f : 0.0f;
    for (int i = 0; i < 64; i++) aux[0 * 64 + i] = color_val;

    // Plane 1: total move count (normalized)
    float move_val = float(pos_.fullmove_number()) / 200.0f; // normalize
    for (int i = 0; i < 64; i++) aux[1 * 64 + i] = move_val;

    // Planes 2-5: castling rights (from current player's perspective)
    CastlingRight cr = pos_.castling();
    float k_oo = (cr & king_side(perspective)) ? 1.0f : 0.0f;
    float k_ooo = (cr & queen_side(perspective)) ? 1.0f : 0.0f;
    float o_oo = (cr & king_side(~perspective)) ? 1.0f : 0.0f;
    float o_ooo = (cr & queen_side(~perspective)) ? 1.0f : 0.0f;
    for (int i = 0; i < 64; i++) {
        aux[2 * 64 + i] = k_oo;
        aux[3 * 64 + i] = k_ooo;
        aux[4 * 64 + i] = o_oo;
        aux[5 * 64 + i] = o_ooo;
    }

    // Plane 6: halfmove clock (normalized)
    float halfmove = float(pos_.halfmove_clock()) / 100.0f;
    for (int i = 0; i < 64; i++) aux[6 * 64 + i] = halfmove;
}

void GameState::get_legal_move_mask(float* mask) const {
    std::memset(mask, 0, POLICY_SIZE * sizeof(float));
    Color perspective = pos_.side_to_move();

    MoveList moves;
    generate_legal(pos_, moves);

    for (int i = 0; i < moves.size(); i++) {
        Move m = moves[i];

        // Handle en passant and castling flags that the encoder needs
        int idx = encode_move(m, perspective);
        if (idx >= 0 && idx < POLICY_SIZE) {
            mask[idx] = 1.0f;
        }
    }
}

} // namespace alphaclaude
