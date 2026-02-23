#pragma once

#include "position.h"
#include "movegen.h"
#include <vector>
#include <deque>
#include <array>
#include <cstring>

namespace alphaclaude {

// ============================================================
// Constants for NN encoding
// ============================================================

constexpr int NUM_HISTORY_STEPS = 8;
constexpr int PLANES_PER_STEP = 14;  // 6 own + 6 opponent + 2 repetition
constexpr int AUX_PLANES = 7;        // color, fullmove, 4 castling, halfmove
constexpr int TOTAL_PLANES = NUM_HISTORY_STEPS * PLANES_PER_STEP + AUX_PLANES; // 119

constexpr int POLICY_SIZE = 4672;     // 8 * 8 * 73

// ============================================================
// GameState: wraps Position with history for NN encoding
// ============================================================

class GameState {
public:
    // Board snapshots for NN encoding (just piece bitboards, no pointer issues)
    struct BoardSnapshot {
        Bitboard by_type[PIECE_TYPE_NB];
        Bitboard by_color[COLOR_NB];
        int repetition;
    };

    GameState();
    GameState(const std::string& fen);
    GameState(const GameState& other);
    GameState& operator=(const GameState& other);
    GameState(GameState&&) = default;
    GameState& operator=(GameState&&) = default;

    // Core operations
    void reset();
    void make_move(Move m);
    void make_move_uci(const std::string& uci);

    // Legal moves
    MoveList legal_moves() const;

    // Game termination
    bool is_terminal() const;
    float terminal_value() const; // +1 if current player won, -1 lost, 0 draw
    bool is_draw() const;

    // NN encoding
    void get_nn_input(float* output) const;

    // Move encoding/decoding
    static int encode_move(Move m, Color perspective);
    static Move decode_move(int idx, Color perspective);

    // Legal move mask (4672 elements)
    void get_legal_move_mask(float* mask) const;

    // Access
    const Position& position() const { return pos_; }
    Color side_to_move() const { return pos_.side_to_move(); }
    int move_count() const { return move_count_; }
    std::string fen() const { return pos_.fen(); }

private:
    void encode_board(const BoardSnapshot& snap, Color perspective, float* planes) const;

    Position pos_;
    std::deque<StateInfo> state_history_;  // deque: no reallocation on push_back
    std::vector<BoardSnapshot> board_history_;
    std::vector<std::string> fen_history_;
    std::vector<std::string> move_history_;
    int move_count_;
};

} // namespace alphaclaude
