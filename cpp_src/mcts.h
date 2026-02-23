#pragma once

#include "game.h"
#include <vector>
#include <array>
#include <memory>
#include <random>
#include <cmath>

namespace alphaclaude {

// ============================================================
// MCTS Configuration
// ============================================================

struct MCTSConfig {
    int num_simulations = 800;
    float c_base = 19652.0f;
    float c_init = 1.25f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    int batch_size = 8;
};

// ============================================================
// MCTS Node
// ============================================================

struct MCTSNode {
    // Tree structure
    MCTSNode* parent = nullptr;
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<Move> child_moves;

    // Statistics
    int visit_count = 0;
    float total_value = 0.0f;
    float prior = 0.0f;

    // State info
    bool is_expanded = false;
    bool is_terminal = false;
    float terminal_value = 0.0f;

    // Game state at this node (lazily set during expansion)
    GameState game_state;

    float q_value() const {
        return visit_count > 0 ? total_value / visit_count : 0.0f;
    }
};

// ============================================================
// MCTS
// ============================================================

class MCTS {
public:
    explicit MCTS(const MCTSConfig& config);

    // Start a new search from the given game state
    void new_search(const GameState& gs);

    // Check if search is complete
    bool search_complete() const;

    // Batched evaluation protocol:
    // 1. get_leaf_batch: select leaves via PUCT, return their NN inputs
    //    Also identifies terminal leaves and their values
    void get_leaf_batch(
        std::vector<std::array<float, TOTAL_PLANES * 64>>& inputs,
        std::vector<std::array<float, POLICY_SIZE>>& masks,
        std::vector<int>& terminal_indices,
        std::vector<float>& terminal_values
    );

    // 2. provide_evaluations: receive NN outputs and expand/backpropagate
    void provide_evaluations(
        const std::vector<float>& values,
        const std::vector<std::array<float, POLICY_SIZE>>& policies
    );

    // After search: get policy target (visit count distribution)
    void get_policy_target(float* output, float temperature) const;

    // Select a move based on visit counts
    Move select_move(float temperature) const;

    // Statistics
    float root_value() const;
    int simulations_done() const { return simulations_done_; }

private:
    MCTSNode* select_leaf(MCTSNode* node);
    float puct_score(MCTSNode* parent, MCTSNode* child) const;
    void backpropagate(MCTSNode* node, float value);
    void expand_node(MCTSNode* node, const std::array<float, POLICY_SIZE>& policy);
    void add_dirichlet_noise(MCTSNode* node);

    MCTSConfig config_;
    std::unique_ptr<MCTSNode> root_;
    int simulations_done_;
    bool noise_added_;

    // Pending leaves for batched evaluation
    std::vector<MCTSNode*> pending_leaves_;

    std::mt19937 rng_;
};

} // namespace alphaclaude
