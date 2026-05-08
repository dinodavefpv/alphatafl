# AlphaTafl: Product Requirements Document (PRD)

## 1. Project Overview
**Project Name:** AlphaTafl  
**Objective:** To develop, train, and deploy an artificial intelligence capable of superhuman gameplay in the asymmetric board game Hnefatafl (Viking Chess). The system will utilize Deep Reinforcement Learning (DRL) and Monte Carlo Tree Search (MCTS), mimicking the *tabula rasa* (from scratch) self-play architecture pioneered by AlphaZero.

## 2. Reference Documents
* **`game_rules.md`**: The source of truth for all game mechanics, board size (11x11), setup states, movement validations, and victory conditions. The core simulator must strictly adhere to these logic constraints.
* **`hnefatafl_gui.py`**: The Python/Pygame-based graphical user interface boilerplate. This serves as the primary visualization tool for observing AI self-play and the interactive client for Human-vs-AI matches.

---

## 3. Core Architectural Decisions

### 3.1 Algorithmic Approach
* **Framework:** Deep Reinforcement Learning with MCTS.
* **Knowledge Acquisition:** 100% Self-Play. The AI will learn entirely through unsupervised reinforcement learning against itself. 
* **Human Data:** Explicitly excluded. No supervised learning on human game databases will be used in order to avoid human bias, suboptimal strategies, and to accelerate processing.

### 3.2 Neural Network Design
* **Architecture:** Convolutional Residual Neural Network (ResNet).
* **Dual-Headed Output:**
  * *Policy Head:* Outputs a probability distribution over all legal moves.
  * *Value Head:* Outputs a continuous scalar [-1, 1] predicting the game's outcome from the current state.
* **Single Network for Both Factions:** Instead of training separate Attacker and Defender models, a single network will be used to maximize shared spatial learning (board geometry, blocking, etc.). 
* **State Representation (Input Tensors):** The board state will be encoded as multi-channel tensors. To handle the game's asymmetry, a dedicated input channel will uniformly flag the current player (e.g., all 1s for Attackers, all 0s for Defenders).

### 3.3 Tech Stack & Hybrid Performance Model
To solve the CPU bottleneck inherent in MCTS and the GPU acceleration required for Deep Learning, a hybrid language approach is adopted:
* **Simulator & MCTS Engine (C++):** All board logic, move generation, and MCTS tree traversals will be written in C++ to guarantee maximum simulation throughput.
* **Neural Network & Training Loop (Python/PyTorch):** The model architecture, backpropagation, and self-play orchestration will be handled in PyTorch.
* **Bridge (Pybind11):** C++ code will be compiled into a Python module to allow seamless real-time data exchange between the Python training loop and the C++ simulation engine.
* **Client UI (Python/Pygame):** The visual layer (`hnefatafl_gui.py`) will import the PyTorch model for inference and the C++ engine for move validation during interactive play.

---

## 4. Hardware & Deployment Targets
The pipeline is designed to run entirely locally without cloud dependencies, optimized for top-tier consumer hardware:
* **CPU (e.g., 24-core / i9-14900K):** Dedicated to running heavily multithreaded MCTS simulations in C++. High thread count is utilized to run dozens of self-play games in parallel.
* **GPU (e.g., 16GB VRAM / RTX 5080):** Dedicated to neural network inference during MCTS and batch training updates. The 16GB VRAM provides ample space for the ResNet weights and large training batches.
* **RAM (e.g., 128GB):** Dedicated to housing the massive Replay Buffer in active memory. This prevents slow SSD read/writes and ensures the PyTorch dataloader can randomly sample hundreds of thousands of historical game states with zero latency.

---

## 5. Phased Implementation Plan

### Phase 1: The Fast Engine (C++)
* Translate `game_rules.md` into a C++ game state class.
* Implement rapid move generation, capture logic (sandwiching), and win-state detection.
* Write unit tests ensuring 100% compliance with `game_rules.md`.
* Wrap the engine using Pybind11.

### Phase 2: The Intuition (PyTorch)
* Design the board state-to-tensor conversion functions.
* Build the ResNet architecture with Policy and Value heads.
* Set up the loss functions (Cross-Entropy for Policy, Mean Squared Error for Value) and the optimizer.

### Phase 3: The Brain & Self-Play (Python + C++)
* Implement the MCTS algorithm in C++ (or highly optimized Python calling the C++ engine).
* Build the Self-Play worker loop: generating games, storing states/probabilities/outcomes in the 128GB RAM Replay Buffer.
* Build the Training Loop: sampling the Replay Buffer, updating network weights, and saving checkpoint models.

### Phase 4: Integration & Visualization
* Connect the finalized `.pt` (PyTorch) model file to `hnefatafl_gui.py`.
* Implement the `request_ai_move()` hook in the GUI to pass the current board state to the AI, run a brief MCTS search using the network, and execute the returned best move on the Pygame board.

## 6. Success Metrics
* **Engine Speed:** Move generation and state resolution > 10,000 states per second per thread.
* **Model Convergence:** The Value loss and Policy loss must steadily decrease over the first 50,000 self-play games.
* **Play Strength:** The AI (using a 1-second search time) defeats a human player with a 100% win rate as both Attackers and Defenders.