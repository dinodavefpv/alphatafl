---
title: Graphical User Interface
---

# Graphical User Interface (GUI)

The `gui.py` script provides a Pygame-based visualizer for AlphaTafl.

## Features

- **Interactive Play**: Play against another human or an AI version.
- **Game Replay**: Load `.json` save files from the `saves/` directory and step through moves.
- **AI Integration**: Uses a trained neural network to provide real-time win probability analysis.
- **Win Probability Chart**: Displays a live graph of the game's predicted outcome (Value head output).
- **Capture Counter**: Tracks the number of pieces captured for each side.
- **Side Selection**: Choose to play as Attackers or Defenders against the AI.
- **AI Thinking Indicator**: Shows "AI is thinking..." while MCTS search is running.

## Controls

- **Mouse Click**: Select and move pieces.
- **Load Game**: Opens the latest save file for replay.
- **Play AI**: Starts a new game against the current best model.
- **Next/Prev**: Navigate through replay moves.
- **Play/Pause**: Automate replay playback.
- **Speed**: Adjust playback speed (1x to 16x).

## Async AI Search (K6, v1)

AI moves are computed on a background `threading.Thread` to keep the GUI responsive:

1. When it's the AI's turn, `ai_is_searching` is set to `True` and a background thread runs `engine.MCTS(eval_fn, 1.4).search(gui.game, 100)` with a local CPU model.
2. The main render loop polls for completion (`ai_search_result`).
3. When the result is ready, the move is applied and the display updates.
4. The GUI uses the **single-leaf** `MCTS::search(state, num_simulations)` path with its own local CPU model copy. It does NOT use the Inference Server, avoiding IPC latency for the ~1 inference/move that the GUI needs.

## Probability Chart Interpretation

- **+1.0**: Strong advantage for Attackers (Black).
- **-1.0**: Strong advantage for Defenders (White).
- **0.0**: Balanced game or Draw.

## Source Files
- [gui.py](../../../gui.py)
