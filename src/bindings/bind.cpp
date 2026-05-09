#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include "game_state.h"
#include "mcts.h"

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
        .def(py::init<MCTS::EvalFn, double>(),
            py::arg("eval_fn"), py::arg("c_puct") = 1.4)
        .def(py::init<MCTS::EvalFn, MCTS::EvalFnBatched, double>(),
            py::arg("eval_fn"), py::arg("eval_fn_batched"), py::arg("c_puct") = 1.4)
        .def("search", static_cast<std::vector<double> (MCTS::*)(const GameState&, int)>(&MCTS::search),
            py::arg("initial_state"), py::arg("num_simulations"))
        .def("search", static_cast<std::vector<double> (MCTS::*)(const GameState&, int, int)>(&MCTS::search),
            py::arg("initial_state"), py::arg("num_simulations"), py::arg("batch_size"));
}

} // namespace alphatafl
