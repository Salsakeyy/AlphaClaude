#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "game.h"
#include "mcts.h"

namespace py = pybind11;

using namespace alphaclaude;

PYBIND11_MODULE(alphaclaude_cpp, m) {
    m.doc() = "AlphaClaude C++ chess engine with MCTS";

    // ============================================================
    // GameState
    // ============================================================
    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("reset", &GameState::reset)
        .def("make_move_uci", &GameState::make_move_uci)
        .def("fen", &GameState::fen)
        .def("side_to_move", [](const GameState& gs) { return int(gs.side_to_move()); })
        .def("move_count", &GameState::move_count)
        .def("is_terminal", &GameState::is_terminal)
        .def("terminal_value", &GameState::terminal_value)
        .def("is_draw", &GameState::is_draw)

        .def("legal_moves_uci", [](const GameState& gs) {
            MoveList moves = gs.legal_moves();
            std::vector<std::string> uci_moves;
            uci_moves.reserve(moves.size());
            for (int i = 0; i < moves.size(); i++)
                uci_moves.push_back(moves[i].to_uci());
            return uci_moves;
        })

        .def("get_nn_input", [](const GameState& gs) {
            py::array_t<float> arr(std::vector<ssize_t>{TOTAL_PLANES, 8, 8});
            auto buf = arr.mutable_data();
            gs.get_nn_input(buf);
            return arr;
        })

        .def("get_legal_move_mask", [](const GameState& gs) {
            py::array_t<float> arr(std::vector<ssize_t>{POLICY_SIZE});
            auto buf = arr.mutable_data();
            gs.get_legal_move_mask(buf);
            return arr;
        })

        .def("encode_move", [](const GameState& gs, const std::string& uci) {
            Move m = gs.position().parse_uci(uci);
            return GameState::encode_move(m, gs.side_to_move());
        })

        .def_static("decode_move_uci", [](int idx, int color) {
            Move m = GameState::decode_move(idx, Color(color));
            return m.to_uci();
        });

    // ============================================================
    // MCTS Config
    // ============================================================
    py::class_<MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("num_simulations", &MCTSConfig::num_simulations)
        .def_readwrite("c_base", &MCTSConfig::c_base)
        .def_readwrite("c_init", &MCTSConfig::c_init)
        .def_readwrite("dirichlet_alpha", &MCTSConfig::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &MCTSConfig::dirichlet_epsilon)
        .def_readwrite("batch_size", &MCTSConfig::batch_size);

    // ============================================================
    // MCTS
    // ============================================================
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const MCTSConfig&>())

        .def("new_search", [](MCTS& mcts, const GameState& gs) {
            mcts.new_search(gs);
        })

        .def("search_complete", &MCTS::search_complete)

        .def("get_leaf_batch", [](MCTS& mcts) {
            std::vector<std::array<float, TOTAL_PLANES * 64>> inputs;
            std::vector<std::array<float, POLICY_SIZE>> masks;
            std::vector<int> terminal_indices;
            std::vector<float> terminal_values;

            mcts.get_leaf_batch(inputs, masks, terminal_indices, terminal_values);

            int n = (int)inputs.size();
            py::array_t<float> py_inputs;
            py::array_t<float> py_masks;

            if (n > 0) {
                py_inputs = py::array_t<float>(std::vector<ssize_t>{n, TOTAL_PLANES, 8, 8});
                py_masks = py::array_t<float>(std::vector<ssize_t>{n, (ssize_t)POLICY_SIZE});
                auto inp_buf = py_inputs.mutable_data();
                auto mask_buf = py_masks.mutable_data();
                for (int i = 0; i < n; i++) {
                    std::memcpy(inp_buf + i * TOTAL_PLANES * 64, inputs[i].data(), TOTAL_PLANES * 64 * sizeof(float));
                    std::memcpy(mask_buf + i * POLICY_SIZE, masks[i].data(), POLICY_SIZE * sizeof(float));
                }
            } else {
                py_inputs = py::array_t<float>(std::vector<ssize_t>{0, TOTAL_PLANES, 8, 8});
                py_masks = py::array_t<float>(std::vector<ssize_t>{0, (ssize_t)POLICY_SIZE});
            }

            return py::make_tuple(py_inputs, py_masks,
                                  py::cast(terminal_indices),
                                  py::cast(terminal_values));
        })

        .def("provide_evaluations", [](MCTS& mcts,
                                        py::array_t<float> values,
                                        py::array_t<float> policies) {
            auto v_buf = values.data();
            auto p_buf = policies.data();
            int n = (int)values.shape(0);

            std::vector<float> vals(v_buf, v_buf + n);
            std::vector<std::array<float, POLICY_SIZE>> pols(n);
            for (int i = 0; i < n; i++) {
                std::memcpy(pols[i].data(), p_buf + i * POLICY_SIZE, POLICY_SIZE * sizeof(float));
            }
            mcts.provide_evaluations(vals, pols);
        })

        .def("get_policy_target", [](MCTS& mcts, float temperature) {
            py::array_t<float> arr(std::vector<ssize_t>{POLICY_SIZE});
            auto buf = arr.mutable_data();
            mcts.get_policy_target(buf, temperature);
            return arr;
        })

        .def("select_move_uci", [](MCTS& mcts, float temperature) {
            Move m = mcts.select_move(temperature);
            return m.to_uci();
        })

        .def("root_value", &MCTS::root_value)
        .def("simulations_done", &MCTS::simulations_done);

    // ============================================================
    // ParallelMCTS
    // ============================================================
    py::class_<ParallelMCTS>(m, "ParallelMCTS")
        .def(py::init<const MCTSConfig&, int>())

        .def("new_search", [](ParallelMCTS& pm, int idx, const GameState& gs) {
            pm.new_search(idx, gs);
        })

        .def("search_complete", &ParallelMCTS::search_complete)
        .def("num_games", &ParallelMCTS::num_games)

        .def("get_all_leaf_batches", [](ParallelMCTS& pm) {
            std::vector<std::array<float, TOTAL_PLANES * 64>> inputs;
            std::vector<std::array<float, POLICY_SIZE>> masks;
            std::vector<int> game_ids;
            std::vector<int> batch_counts;

            pm.get_all_leaf_batches(inputs, masks, game_ids, batch_counts);

            int n = (int)inputs.size();
            py::array_t<float> py_inputs;
            py::array_t<float> py_masks;

            if (n > 0) {
                py_inputs = py::array_t<float>(std::vector<ssize_t>{n, TOTAL_PLANES, 8, 8});
                py_masks = py::array_t<float>(std::vector<ssize_t>{n, (ssize_t)POLICY_SIZE});
                auto inp_buf = py_inputs.mutable_data();
                auto mask_buf = py_masks.mutable_data();
                for (int i = 0; i < n; i++) {
                    std::memcpy(inp_buf + i * TOTAL_PLANES * 64,
                                inputs[i].data(), TOTAL_PLANES * 64 * sizeof(float));
                    std::memcpy(mask_buf + i * POLICY_SIZE,
                                masks[i].data(), POLICY_SIZE * sizeof(float));
                }
            } else {
                py_inputs = py::array_t<float>(std::vector<ssize_t>{0, TOTAL_PLANES, 8, 8});
                py_masks = py::array_t<float>(std::vector<ssize_t>{0, (ssize_t)POLICY_SIZE});
            }

            py::array_t<int> py_game_ids(game_ids.size());
            py::array_t<int> py_batch_counts(batch_counts.size());
            if (!game_ids.empty()) {
                std::memcpy(py_game_ids.mutable_data(), game_ids.data(),
                            game_ids.size() * sizeof(int));
                std::memcpy(py_batch_counts.mutable_data(), batch_counts.data(),
                            batch_counts.size() * sizeof(int));
            }

            return py::make_tuple(py_inputs, py_masks, py_game_ids, py_batch_counts);
        })

        .def("provide_all_evaluations", [](ParallelMCTS& pm,
                                            py::array_t<float> values,
                                            py::array_t<float> policies,
                                            py::array_t<int> game_ids,
                                            py::array_t<int> batch_counts) {
            pm.provide_all_evaluations(
                values.data(),
                policies.data(),
                game_ids.data(),
                batch_counts.data(),
                (int)game_ids.shape(0)
            );
        })

        .def("get_policy_target", [](const ParallelMCTS& pm, int idx, float temperature) {
            py::array_t<float> arr(std::vector<ssize_t>{POLICY_SIZE});
            auto buf = arr.mutable_data();
            pm.get_policy_target(idx, buf, temperature);
            return arr;
        })

        .def("select_move_uci", [](const ParallelMCTS& pm, int idx, float temperature) {
            Move m = pm.select_move(idx, temperature);
            return m.to_uci();
        })

        .def("reset_game", &ParallelMCTS::reset_game);

    // ============================================================
    // Constants
    // ============================================================
    m.attr("TOTAL_PLANES") = TOTAL_PLANES;
    m.attr("POLICY_SIZE") = POLICY_SIZE;
}
