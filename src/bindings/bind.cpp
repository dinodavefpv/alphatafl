#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include "game_state.h"
#include "mcts.h"
#include "inference_engine.h"

namespace py = pybind11;

namespace alphatafl {

PYBIND11_MODULE(alphatafl_engine, m) {
    m.doc() = "AlphaTafl C++ Engine Bindings";

    py::enum_<Piece>(m, "Piece")
        .value("EMPTY", Piece::EMPTY)
        .value("ATTACKER", Piece::ATTACKER)
        .value("DEFENDER", Piece::DEFENDER)
        .value("KING", Piece::KING)
        .export_values();

    py::enum_<Player>(m, "Player")
        .value("NONE", Player::NONE)
        .value("ATTACKER", Player::ATTACKER)
        .value("DEFENDER", Player::DEFENDER)
        .value("DRAW", Player::DRAW)
        .export_values();

    py::class_<Move>(m, "Move")
        .def(py::init<int, int, int, int>())
        .def_readwrite("from_row", &Move::from_row)
        .def_readwrite("from_col", &Move::from_col)
        .def_readwrite("to_row", &Move::to_row)
        .def_readwrite("to_col", &Move::to_col)
        .def(py::self == py::self);

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def("reset", &GameState::reset)
        .def("get_legal_moves", &GameState::get_legal_moves)
        .def("apply_move", &GameState::apply_move)
        .def("get_piece", &GameState::get_piece)
        .def("clone", &GameState::clone)
        .def("to_tensor", [](const GameState& s) {
            auto v = s.to_tensor();
            float* buf = new float[v.size()];
            std::copy(v.begin(), v.end(), buf);
            return py::array_t<float>(
                {14, 11, 11},
                {11*11*4, 11*4, 4},
                buf,
                py::capsule(buf, [](void* p) { delete[] static_cast<float*>(p); })
            );
        })
        .def_static("batch_to_tensor", [](const std::vector<GameState>& states) {
            auto v = GameState::batch_to_tensor(states);
            float* buf = new float[v.size()];
            std::copy(v.begin(), v.end(), buf);
            long n = static_cast<long>(states.size());
            std::vector<long> shape = {n, 14, 11, 11};
            std::vector<long> strides = {14*11*11*4L, 11*11*4L, 11*4L, 4L};
            return py::array_t<float>(shape, strides, buf,
                py::capsule(buf, [](void* p) { delete[] static_cast<float*>(p); }));
        })
        .def("get_legal_moves_mask", [](const GameState& s) {
            auto v = s.get_legal_moves_mask();
            float* buf = new float[v.size()];
            std::copy(v.begin(), v.end(), buf);
            std::vector<long> shape = {static_cast<long>(v.size())};
            std::vector<long> strides = {4L};
            return py::array_t<float>(shape, strides, buf,
                py::capsule(buf, [](void* p) { delete[] static_cast<float*>(p); }));
        })
        .def_readwrite("current_turn", &GameState::current_turn)
        .def_readwrite("winner", &GameState::winner);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init([](py::function eval_fn, double c_puct, double dirichlet_alpha, double dirichlet_epsilon) {
            auto cpp_eval_fn = [eval_fn](const GameState& state) -> std::pair<std::vector<float>, float> {
                py::gil_scoped_acquire acquire;
                py::tuple result = eval_fn(state);
                
                py::array_t<float> policy_arr;
                if (py::isinstance<py::array_t<float>>(result[0])) {
                    policy_arr = result[0].cast<py::array_t<float>>();
                } else {
                    py::object np = py::module_::import("numpy");
                    py::object np_arr = np.attr("array")(result[0], py::arg("dtype") = "float32");
                    policy_arr = np_arr.cast<py::array_t<float>>();
                }
                
                auto policy_buf = policy_arr.request();
                float* policy_ptr = static_cast<float*>(policy_buf.ptr);
                
                std::vector<float> policy(policy_ptr, policy_ptr + 4840);
                float value = result[1].cast<float>();
                return {policy, value};
            };
            return std::make_unique<MCTS>(cpp_eval_fn, c_puct, dirichlet_alpha, dirichlet_epsilon);
        }), py::arg("eval_fn"), py::arg("c_puct") = 1.4, py::arg("dirichlet_alpha") = 0.3, py::arg("dirichlet_epsilon") = 0.0)
        .def(py::init([](py::function eval_fn, py::object eval_fn_batched_obj, double c_puct, double dirichlet_alpha, double dirichlet_epsilon) {
            auto cpp_eval_fn = [eval_fn](const GameState& state) -> std::pair<std::vector<float>, float> {
                py::gil_scoped_acquire acquire;
                py::tuple result = eval_fn(state);
                
                py::array_t<float> policy_arr;
                if (py::isinstance<py::array_t<float>>(result[0])) {
                    policy_arr = result[0].cast<py::array_t<float>>();
                } else {
                    py::object np = py::module_::import("numpy");
                    py::object np_arr = np.attr("array")(result[0], py::arg("dtype") = "float32");
                    policy_arr = np_arr.cast<py::array_t<float>>();
                }
                
                auto policy_buf = policy_arr.request();
                float* policy_ptr = static_cast<float*>(policy_buf.ptr);
                
                std::vector<float> policy(policy_ptr, policy_ptr + 4840);
                float value = result[1].cast<float>();
                return {policy, value};
            };
            
            if (eval_fn_batched_obj.is_none()) {
                return std::make_unique<MCTS>(cpp_eval_fn, c_puct, dirichlet_alpha, dirichlet_epsilon);
            }
            
            py::function eval_fn_batched = eval_fn_batched_obj.cast<py::function>();
            
            auto cpp_eval_fn_batched = [eval_fn_batched](const std::vector<GameState>& states) -> std::pair<std::vector<float>, std::vector<float>> {
                py::gil_scoped_acquire acquire;
                py::tuple result = eval_fn_batched(states);
                
                py::array_t<float> policies_arr;
                if (py::isinstance<py::array_t<float>>(result[0])) {
                    policies_arr = result[0].cast<py::array_t<float>>();
                } else {
                    py::object np = py::module_::import("numpy");
                    py::object np_arr = np.attr("array")(result[0], py::arg("dtype") = "float32");
                    policies_arr = np_arr.cast<py::array_t<float>>();
                }
                
                py::array_t<float> values_arr;
                if (py::isinstance<py::array_t<float>>(result[1])) {
                    values_arr = result[1].cast<py::array_t<float>>();
                } else {
                    py::object np = py::module_::import("numpy");
                    py::object np_arr = np.attr("array")(result[1], py::arg("dtype") = "float32");
                    values_arr = np_arr.cast<py::array_t<float>>();
                }
                
                auto policies_buf = policies_arr.request();
                auto values_buf = values_arr.request();
                
                float* policies_ptr = static_cast<float*>(policies_buf.ptr);
                float* values_ptr = static_cast<float*>(values_buf.ptr);
                
                size_t num_states = states.size();
                
                std::vector<float> flat_policies(policies_ptr, policies_ptr + num_states * 4840);
                std::vector<float> flat_values(values_ptr, values_ptr + num_states);
                
                return {flat_policies, flat_values};
            };
            
            return std::make_unique<MCTS>(cpp_eval_fn, cpp_eval_fn_batched, c_puct, dirichlet_alpha, dirichlet_epsilon);
        }), py::arg("eval_fn"), py::arg("eval_fn_batched"), py::arg("c_puct") = 1.4, py::arg("dirichlet_alpha") = 0.3, py::arg("dirichlet_epsilon") = 0.0)
        .def("search", static_cast<std::vector<float> (MCTS::*)(const GameState&, int)>(&MCTS::search),
            py::arg("initial_state"), py::arg("num_simulations"))
        .def("search", static_cast<std::vector<float> (MCTS::*)(const GameState&, int, int)>(&MCTS::search),
            py::arg("initial_state"), py::arg("num_simulations"), py::arg("batch_size"));

    // Phase 5B: Native C++ inference engine (eliminates Python mp.Queue)
    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<const std::string&, const std::string&>(),
            py::arg("model_path"), py::arg("device") = "cuda")
        .def("evaluate", &InferenceEngine::evaluate,
            py::arg("states"),
            "Evaluate a batch of GameStates. Returns (policies, values).")
        .def("evaluate_single", &InferenceEngine::evaluate_single,
            py::arg("state"),
            "Evaluate a single GameState. Returns (policy, value).");
}

} // namespace alphatafl
