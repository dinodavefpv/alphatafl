---
title: ResNet Architecture
---

# Neural Network Architecture

AlphaTafl uses a Convolutional Residual Neural Network (ResNet) to provide "intuition" for the MCTS search.

## Overview

The network takes a representation of the 11x11 board as input and produces two outputs:
1.  **Policy Head**: A probability distribution over all 4840 possible moves.
2.  **Value Head**: A scalar value in the range `[-1, 1]` representing the predicted outcome for the current player.

## Input Representation

The board state is converted into a 14-channel tensor of shape `(14, 11, 11)`:

- **Channels 0-2** (T): Attacker, Defender, King positions on the current board.
- **Channels 3-5** (T-1): Same, one move ago.
- **Channels 6-8** (T-2): Same, two moves ago.
- **Channels 9-11** (T-3): Same, three moves ago.
- **Channel 12**: Turn indicator (1.0 if Attacker's turn, 0.0 if Defender's).
- **Channel 13**: Restricted squares mask (corners + throne).

## Architecture Details

- **Residual Blocks**: Multiple blocks consisting of convolutional layers, batch normalization, and skip connections. This allows for deep networks without vanishing gradients.
- **Policy Head**:
    - Batch normalization.
    - ReLU activation.
    - Convolution (1x1) with 40 filters (each output channel corresponds to one of 40 direction-distance combinations per square).
    - Permute + reshape to flat `(batch, 4840)` logits.
- **Value Head**:
    - Convolution (1x1) with 1 filter.
    - Batch normalization.
    - ReLU activation.
    - Flatten to 121.
    - Fully connected layer: 121 -> 128.
    - ReLU activation.
    - Fully connected layer: 128 -> 1 with Tanh activation.

## Training Objective

The loss function $L$ is a combination of Mean Squared Error (MSE) for the value head and Cross-Entropy for the policy head:

$L = (z - v)^2 - \pi \log p$

- $z$: Actual game outcome from self-play (-1, 0, or +1).
- $v$: Predicted value (output of Tanh, range [-1, 1]).
- $\pi$: MCTS search probabilities (target distribution).
- $p$: Predicted policy (logits passed through log-softmax).

L2 regularization is applied implicitly via Adam's `weight_decay=1e-4` in the optimizer, rather than as an explicit term in the loss function.

## Source Files
- [network.py](../../../src/model/network.py)
