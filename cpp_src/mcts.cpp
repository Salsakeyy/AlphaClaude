#include "mcts.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>

namespace alphaclaude {

MCTS::MCTS(const MCTSConfig& config)
    : config_(config), simulations_done_(0), noise_added_(false),
      rng_(std::random_device{}()) {}

void MCTS::new_search(const GameState& gs) {
    root_ = std::make_unique<MCTSNode>();
    root_->game_state = gs;
    simulations_done_ = 0;
    noise_added_ = false;
    pending_leaves_.clear();
}

bool MCTS::search_complete() const {
    return simulations_done_ >= config_.num_simulations;
}

float MCTS::puct_score(MCTSNode* parent, MCTSNode* child) const {
    float c = std::log((1.0f + parent->visit_count + config_.c_base) / config_.c_base) + config_.c_init;
    float u = c * child->prior * std::sqrt(float(parent->visit_count)) / (1.0f + child->visit_count);
    // Q from parent's perspective: negate child's Q since it's from opponent's view
    float q = child->visit_count > 0 ? -child->q_value() : 0.0f;
    return q + u;
}

MCTSNode* MCTS::select_leaf(MCTSNode* node) {
    while (node->is_expanded && !node->is_terminal) {
        // Select child with highest PUCT score
        MCTSNode* best = nullptr;
        float best_score = -1e9f;

        for (auto& child : node->children) {
            float score = puct_score(node, child.get());
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }

        if (!best) break;
        node = best;
    }
    return node;
}

void MCTS::backpropagate(MCTSNode* node, float value) {
    // value is from the perspective of the player at this node
    while (node) {
        node->visit_count++;
        node->total_value += value;
        value = -value; // flip for parent (opponent's perspective)
        node = node->parent;
    }
}

void MCTS::expand_node(MCTSNode* node, const std::array<float, POLICY_SIZE>& policy) {
    if (node->is_expanded || node->is_terminal) return;

    MoveList moves = node->game_state.legal_moves();

    if (moves.size() == 0) {
        node->is_terminal = true;
        node->terminal_value = node->game_state.terminal_value();
        node->is_expanded = true;
        return;
    }

    if (node->game_state.is_terminal()) {
        node->is_terminal = true;
        node->terminal_value = node->game_state.terminal_value();
        node->is_expanded = true;
        return;
    }

    Color perspective = node->game_state.side_to_move();

    // Apply mask and softmax to get priors
    std::vector<float> priors(moves.size());
    float max_logit = -1e9f;
    for (int i = 0; i < moves.size(); i++) {
        int idx = GameState::encode_move(moves[i], perspective);
        if (idx >= 0 && idx < POLICY_SIZE)
            priors[i] = policy[idx];
        else
            priors[i] = -1e9f;
        max_logit = std::max(max_logit, priors[i]);
    }

    float sum = 0;
    for (int i = 0; i < moves.size(); i++) {
        priors[i] = std::exp(priors[i] - max_logit);
        sum += priors[i];
    }
    if (sum > 0) {
        for (int i = 0; i < moves.size(); i++)
            priors[i] /= sum;
    } else {
        float uniform = 1.0f / moves.size();
        for (int i = 0; i < moves.size(); i++)
            priors[i] = uniform;
    }

    // Create children
    node->children.resize(moves.size());
    node->child_moves.resize(moves.size());

    for (int i = 0; i < moves.size(); i++) {
        auto child = std::make_unique<MCTSNode>();
        child->parent = node;
        child->prior = priors[i];
        child->game_state = node->game_state;
        child->game_state.make_move(moves[i]);
        node->child_moves[i] = moves[i];
        node->children[i] = std::move(child);
    }

    node->is_expanded = true;
}

void MCTS::add_dirichlet_noise(MCTSNode* node) {
    if (noise_added_ || !node->is_expanded || node->children.empty()) return;

    std::gamma_distribution<float> gamma(config_.dirichlet_alpha, 1.0f);
    std::vector<float> noise(node->children.size());
    float sum = 0;
    for (size_t i = 0; i < noise.size(); i++) {
        noise[i] = gamma(rng_);
        sum += noise[i];
    }
    if (sum > 0) {
        for (size_t i = 0; i < noise.size(); i++) {
            noise[i] /= sum;
            node->children[i]->prior = (1.0f - config_.dirichlet_epsilon) * node->children[i]->prior
                                     + config_.dirichlet_epsilon * noise[i];
        }
    }
    noise_added_ = true;
}

void MCTS::get_leaf_batch(
    std::vector<std::array<float, TOTAL_PLANES * 64>>& inputs,
    std::vector<std::array<float, POLICY_SIZE>>& masks,
    std::vector<int>& terminal_indices,
    std::vector<float>& terminal_values)
{
    inputs.clear();
    masks.clear();
    terminal_indices.clear();
    terminal_values.clear();
    pending_leaves_.clear();

    int batch = std::min(config_.batch_size, config_.num_simulations - simulations_done_);

    for (int i = 0; i < batch; i++) {
        MCTSNode* leaf = select_leaf(root_.get());

        if (leaf->is_terminal) {
            // Terminal node: directly backpropagate
            backpropagate(leaf, leaf->terminal_value);
            simulations_done_++;
            i--; // don't count towards batch
            batch = std::min(batch, config_.num_simulations - simulations_done_);
            if (batch <= 0) break;
            continue;
        }

        // Collect NN input
        std::array<float, TOTAL_PLANES * 64> input;
        std::array<float, POLICY_SIZE> mask;
        std::memset(input.data(), 0, sizeof(input));
        std::memset(mask.data(), 0, sizeof(mask));

        leaf->game_state.get_nn_input(input.data());
        leaf->game_state.get_legal_move_mask(mask.data());

        inputs.push_back(input);
        masks.push_back(mask);
        pending_leaves_.push_back(leaf);

        // Add virtual loss to prevent re-selecting same leaf
        leaf->visit_count++;
        leaf->total_value -= 1.0f; // pessimistic
    }
}

void MCTS::provide_evaluations(
    const std::vector<float>& values,
    const std::vector<std::array<float, POLICY_SIZE>>& policies)
{
    for (size_t i = 0; i < pending_leaves_.size(); i++) {
        MCTSNode* leaf = pending_leaves_[i];

        // Remove virtual loss
        leaf->visit_count--;
        leaf->total_value += 1.0f;

        // Expand the node
        expand_node(leaf, policies[i]);

        // Add Dirichlet noise to root after first expansion
        if (leaf == root_.get()) {
            add_dirichlet_noise(leaf);
        }

        // Backpropagate value
        float value;
        if (leaf->is_terminal) {
            value = leaf->terminal_value;
        } else {
            value = values[i];
        }
        backpropagate(leaf, value);
        simulations_done_++;
    }
    pending_leaves_.clear();
}

void MCTS::get_policy_target(float* output, float temperature) const {
    std::memset(output, 0, POLICY_SIZE * sizeof(float));

    if (!root_ || !root_->is_expanded) return;

    Color perspective = root_->game_state.side_to_move();

    if (temperature < 0.05f) {
        // Near-deterministic: pick the move with the most visits
        int best_idx = 0;
        int best_visits = 0;
        for (size_t i = 0; i < root_->children.size(); i++) {
            if (root_->children[i]->visit_count > best_visits) {
                best_visits = root_->children[i]->visit_count;
                best_idx = i;
            }
        }
        int move_idx = GameState::encode_move(root_->child_moves[best_idx], perspective);
        if (move_idx >= 0 && move_idx < POLICY_SIZE)
            output[move_idx] = 1.0f;
    } else {
        // Temperature-scaled visit counts
        // Use log-space to avoid overflow: log(p_i) = (1/temp) * log(visit_count_i)
        float inv_temp = 1.0f / temperature;
        std::vector<float> log_probs(root_->children.size());
        float max_log = -1e9f;
        for (size_t i = 0; i < root_->children.size(); i++) {
            float vc = float(root_->children[i]->visit_count);
            log_probs[i] = (vc > 0) ? inv_temp * std::log(vc) : -1e9f;
            max_log = std::max(max_log, log_probs[i]);
        }
        float sum = 0;
        std::vector<float> probs(root_->children.size());
        for (size_t i = 0; i < root_->children.size(); i++) {
            probs[i] = std::exp(log_probs[i] - max_log);
            sum += probs[i];
        }
        if (sum > 0) {
            for (size_t i = 0; i < root_->children.size(); i++) {
                int move_idx = GameState::encode_move(root_->child_moves[i], perspective);
                if (move_idx >= 0 && move_idx < POLICY_SIZE)
                    output[move_idx] = probs[i] / sum;
            }
        }
    }
}

Move MCTS::select_move(float temperature) const {
    if (!root_ || !root_->is_expanded || root_->children.empty())
        return MOVE_NONE;

    if (temperature < 0.05f) {
        // Deterministic: pick highest visit count
        int best_idx = 0;
        int best_visits = 0;
        for (size_t i = 0; i < root_->children.size(); i++) {
            if (root_->children[i]->visit_count > best_visits) {
                best_visits = root_->children[i]->visit_count;
                best_idx = i;
            }
        }
        return root_->child_moves[best_idx];
    } else {
        // Stochastic: sample proportional to visit^(1/temp) using log-space
        float inv_temp = 1.0f / temperature;
        std::vector<float> log_probs(root_->children.size());
        float max_log = -1e9f;
        for (size_t i = 0; i < root_->children.size(); i++) {
            float vc = float(root_->children[i]->visit_count);
            log_probs[i] = (vc > 0) ? inv_temp * std::log(vc) : -1e9f;
            max_log = std::max(max_log, log_probs[i]);
        }
        std::vector<float> probs(root_->children.size());
        for (size_t i = 0; i < root_->children.size(); i++) {
            probs[i] = std::exp(log_probs[i] - max_log);
        }

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int idx = dist(const_cast<std::mt19937&>(rng_));
        return root_->child_moves[idx];
    }
}

float MCTS::root_value() const {
    if (!root_ || root_->visit_count == 0) return 0.0f;
    return root_->q_value();
}

// ============================================================
// ParallelMCTS implementation
// ============================================================

ParallelMCTS::ParallelMCTS(const MCTSConfig& config, int num_games)
    : config_(config), num_games_(num_games) {
    trees_.reserve(num_games);
    for (int i = 0; i < num_games; i++) {
        trees_.push_back(std::make_unique<MCTS>(config));
    }
}

void ParallelMCTS::new_search(int game_idx, const GameState& gs) {
    trees_[game_idx]->new_search(gs);
}

bool ParallelMCTS::search_complete(int game_idx) const {
    return trees_[game_idx]->search_complete();
}

void ParallelMCTS::get_all_leaf_batches(
    std::vector<std::array<float, TOTAL_PLANES * 64>>& inputs,
    std::vector<std::array<float, POLICY_SIZE>>& masks,
    std::vector<int>& game_ids,
    std::vector<int>& batch_counts)
{
    inputs.clear();
    masks.clear();
    game_ids.clear();
    batch_counts.clear();

    // Reserve estimated capacity to reduce reallocations
    int active_count = 0;
    for (int g = 0; g < num_games_; g++) {
        if (!trees_[g]->search_complete()) active_count++;
    }
    inputs.reserve(active_count * config_.batch_size);
    masks.reserve(active_count * config_.batch_size);
    game_ids.reserve(active_count);
    batch_counts.reserve(active_count);

    // Temporary per-tree buffers (reused to avoid reallocation)
    std::vector<std::array<float, TOTAL_PLANES * 64>> tree_inputs;
    std::vector<std::array<float, POLICY_SIZE>> tree_masks;
    std::vector<int> tree_terminal_indices;
    std::vector<float> tree_terminal_values;

    for (int g = 0; g < num_games_; g++) {
        if (trees_[g]->search_complete()) continue;

        trees_[g]->get_leaf_batch(tree_inputs, tree_masks,
                                  tree_terminal_indices, tree_terminal_values);

        int n = (int)tree_inputs.size();
        if (n > 0) {
            game_ids.push_back(g);
            batch_counts.push_back(n);
            for (int i = 0; i < n; i++) {
                inputs.push_back(std::move(tree_inputs[i]));
                masks.push_back(std::move(tree_masks[i]));
            }
        }
    }
}

void ParallelMCTS::provide_all_evaluations(
    const float* values,
    const float* policies,
    const int* game_ids,
    const int* batch_counts,
    int num_games_in_batch)
{
    int offset = 0;
    for (int i = 0; i < num_games_in_batch; i++) {
        int g = game_ids[i];
        int count = batch_counts[i];

        std::vector<float> vals(values + offset, values + offset + count);
        std::vector<std::array<float, POLICY_SIZE>> pols(count);
        for (int j = 0; j < count; j++) {
            std::memcpy(pols[j].data(),
                        policies + (offset + j) * POLICY_SIZE,
                        POLICY_SIZE * sizeof(float));
        }

        trees_[g]->provide_evaluations(vals, pols);
        offset += count;
    }
}

void ParallelMCTS::get_policy_target(int game_idx, float* output, float temperature) const {
    trees_[game_idx]->get_policy_target(output, temperature);
}

Move ParallelMCTS::select_move(int game_idx, float temperature) const {
    return trees_[game_idx]->select_move(temperature);
}

void ParallelMCTS::reset_game(int game_idx) {
    trees_[game_idx] = std::make_unique<MCTS>(config_);
}

} // namespace alphaclaude
