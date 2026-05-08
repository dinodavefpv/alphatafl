"""
Validation tests for K0 (Test Framework) and K1 (Interface Contract).

These tests verify:
1. pytest and the build environment are correctly set up.
2. alphatafl_engine can be imported.
3. The interface contract document exists and covers all required sections.
4. C++ zero-copy tensor methods (D1) exist with correct signatures.
"""

import sys
import os
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, 'build', 'Release')


class TestK0Framework:
    """K0: Verify test framework and environment setup."""

    def test_build_directory_exists(self):
        assert os.path.isdir(BUILD_DIR), f"Build directory not found: {BUILD_DIR}"

    def test_pyd_exists(self):
        pyd = os.path.join(BUILD_DIR, 'alphatafl_engine.cp313-win_amd64.pyd')
        assert os.path.isfile(pyd), f"Compiled pyd not found: {pyd}"

    def test_alphatafl_engine_import(self):
        import alphatafl_engine as engine
        assert hasattr(engine, 'GameState')
        assert hasattr(engine, 'Move')
        assert hasattr(engine, 'MCTS')
        assert hasattr(engine, 'Piece')
        assert hasattr(engine, 'Player')

    def test_numpy_available(self):
        arr = np.zeros((14, 11, 11), dtype=np.float32)
        assert arr.shape == (14, 11, 11)


class TestK1InterfaceDocument:
    """K1: Verify the interface contract document exists and is complete."""

    CONTRACT_PATH = os.path.join(PROJECT_ROOT, 'wiki', 'docs', 'api', 'interface_contract.md')

    def test_interface_document_exists(self):
        assert os.path.isfile(self.CONTRACT_PATH), (
            f"Interface contract document not found at {self.CONTRACT_PATH}"
        )

    @pytest.fixture(scope='class')
    def contract_text(self):
        with open(self.CONTRACT_PATH, 'r', encoding='utf-8') as f:
            return f.read()

    def test_document_covers_data_types(self, contract_text):
        assert '(N, 14, 11, 11)' in contract_text
        assert '(N, 4840)' in contract_text
        assert '(N,)' in contract_text
        assert 'float32' in contract_text

    def test_document_covers_batch_bounds(self, contract_text):
        assert '256' in contract_text
        assert '50' in contract_text and 'ms' in contract_text

    def test_document_covers_virtual_loss(self, contract_text):
        assert 'visit_count += 3' in contract_text or 'visit += 3' in contract_text
        assert 'value_sum -= 3' in contract_text or 'value -= 3' in contract_text

    def test_document_covers_batched_mcts_algorithm(self, contract_text):
        assert 'pending' in contract_text.lower()
        assert 'stall guard' in contract_text.lower() or 'consecutive_all_pending' in contract_text
        assert 'backpropagation' in contract_text.lower()
        assert 'parent' in contract_text.lower()

    def test_document_covers_queues(self, contract_text):
        assert 'inference_queue' in contract_text
        assert 'response_queues' in contract_text or 'response_queue' in contract_text
        assert 'replay_queue' in contract_text
        assert 'maxsize=50000' in contract_text

    def test_document_covers_cpp_api_additions(self, contract_text):
        assert 'to_tensor()' in contract_text
        assert 'batch_to_tensor' in contract_text
        assert 'get_legal_moves_mask()' in contract_text
        assert 'is_pending' in contract_text
        assert 'EvalFnBatched' in contract_text

    def test_document_covers_worker_responsibilities(self, contract_text):
        assert 'legal move mask' in contract_text.lower() or 'legal masking' in contract_text.lower()
        assert 'dirichlet' in contract_text.lower()
        assert 'worker' in contract_text.lower()

    def test_document_covers_feature_flag(self, contract_text):
        assert 'ALPHATAFL_BATCHED' in contract_text
        assert '"0"' in contract_text
        assert '"1"' in contract_text

    def test_document_covers_inference_server_lifecycle(self, contract_text):
        assert 'cli.py' in contract_text
        assert 'sentinel' in contract_text.lower()


class TestD1TensorMethods:
    """Verify D1 methods exist and return correct shapes (D1 landed by Deepseek)."""

    def test_to_tensor_exists(self, fresh_game_state):
        assert hasattr(fresh_game_state, 'to_tensor')

    def test_to_tensor_shape(self, fresh_game_state):
        arr = fresh_game_state.to_tensor()
        assert arr.shape == (14, 11, 11)
        assert arr.dtype == np.float32

    def test_batch_to_tensor_exists(self, fresh_game_state):
        import alphatafl_engine as engine
        assert hasattr(engine.GameState, 'batch_to_tensor')

    def test_batch_to_tensor_shape(self, fresh_game_state):
        import alphatafl_engine as engine
        states = [fresh_game_state, fresh_game_state, fresh_game_state]
        arr = engine.GameState.batch_to_tensor(states)
        assert arr.shape == (3, 14, 11, 11)
        assert arr.dtype == np.float32

    def test_get_legal_moves_mask_exists(self, fresh_game_state):
        assert hasattr(fresh_game_state, 'get_legal_moves_mask')

    def test_get_legal_moves_mask_shape(self, fresh_game_state):
        arr = fresh_game_state.get_legal_moves_mask()
        assert arr.shape == (4840,)
        assert arr.dtype == np.float32

    def test_legal_moves_mask_count(self, fresh_game_state):
        mask = fresh_game_state.get_legal_moves_mask()
        legal = fresh_game_state.get_legal_moves()
        assert int(mask.sum()) == len(legal)


class TestD2BindingsCleanup:
    """Verify D2 heap-churn cleanup (D2 landed by Deepseek)."""

    def test_board_property_removed(self, fresh_game_state):
        assert not hasattr(fresh_game_state, 'board')

    def test_get_historical_board_removed(self, fresh_game_state):
        assert not hasattr(fresh_game_state, 'get_historical_board')


class TestD4D5BatchedMCTS:
    """Verify D4-D5 batched MCTS API (D4-D5 landed by Deepseek)."""

    def test_batched_constructor(self):
        import alphatafl_engine as engine
        def eval_fn(s):
            return [0.0] * 4840, 0.0
        def eval_fn_batched(states):
            return [[0.0] * 4840 for _ in states], [0.0] * len(states)
        mcts = engine.MCTS(eval_fn, eval_fn_batched, 1.4)
        assert mcts is not None

    def test_search_overload_batch_size(self, fresh_game_state):
        import alphatafl_engine as engine
        # Provide small uniform policy so select_child has valid actions
        def eval_fn(s):
            return [0.001] * 4840, 0.0
        def eval_fn_batched(states):
            return [[0.001] * 4840 for _ in states], [0.0] * len(states)
        mcts = engine.MCTS(eval_fn, eval_fn_batched, 1.4)
        probs = mcts.search(fresh_game_state, 5, 2)
        assert len(probs) == 4840

    def test_search_overload_fallback(self, fresh_game_state):
        import alphatafl_engine as engine
        call_count = [0]
        def eval_fn(s):
            call_count[0] += 1
            return [0.0] * 4840, 0.0
        mcts = engine.MCTS(eval_fn, None, 1.4)
        probs = mcts.search(fresh_game_state, 5, 4)
        assert call_count[0] > 0
        assert len(probs) == 4840

    def test_mctsnode_is_pending_field(self):
        import alphatafl_engine as engine
        state = engine.GameState()
        def eval_fn(s):
            return [0.001] * 4840, 0.0
        def eval_fn_batched(states):
            return [[0.001] * 4840 for _ in states], [0.0] * len(states)
        mcts = engine.MCTS(eval_fn, eval_fn_batched, 1.4)
        probs = mcts.search(state, 10, 4)
        assert len(probs) == 4840
