#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
        .def_readwrite("current_turn", &GameState::current_turn)
        .def_readwrite("winner", &GameState::winner)
        .def_readwrite("history_hashes", &GameState::history_hashes)
        .def_property_readonly("board", [](const GameState& s) {
            std::vector<std::vector<Piece>> b(BOARD_SIZE, std::vector<Piece>(BOARD_SIZE));
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    b[r][c] = s.board[r][c];
                }
            }
            return b;
        })
        .def("get_historical_board", [](const GameState& s, int depth) {
            std::vector<std::vector<Piece>> b(BOARD_SIZE, std::vector<Piece>(BOARD_SIZE));
            auto hist_board = s.get_historical_board(depth);
            for (int r = 0; r < BOARD_SIZE; ++r) {
                for (int c = 0; c < BOARD_SIZE; ++c) {
                    b[r][c] = hist_board[r][c];
                }
            }
            return b;
        }, py::arg("depth"));

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<MCTS::EvalFn, double>(), py::arg("eval_fn"), py::arg("c_puct") = 1.4)
        .def("search", &MCTS::search, py::arg("initial_state"), py::arg("num_simulations"));
}

} // namespace alphatafl
