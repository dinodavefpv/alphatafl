import sys
import os
import numpy as np
import pytest

# Ensure the project root and build directory are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, 'build', 'Release')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)


@pytest.fixture
def fresh_game_state():
    """Return a freshly reset GameState."""
    import alphatafl_engine as engine
    state = engine.GameState()
    state.reset()
    return state


@pytest.fixture
def deterministic_seed():
    """Set a fixed numpy random seed for deterministic tests."""
    np.random.seed(42)
    yield
    np.random.seed(None)
