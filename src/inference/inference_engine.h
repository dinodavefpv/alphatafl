#pragma once

#include <string>
#include <vector>
#include <torch/script.h>
#include "game_state.h"

namespace alphatafl {

/**
 * Native C++ inference engine using TorchScript.
 * Eliminates Python mp.Queue round-trips by running GPU inference
 * directly inside the C++ extension.
 */
class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path, const std::string& device = "cuda");

    /**
     * Evaluate a batch of game states.
     * Returns: (policies, values) where policies is [N][4840] and values is [N].
     */
    std::pair<std::vector<std::vector<float>>, std::vector<float>>
    evaluate(const std::vector<GameState>& states);

    /**
     * Evaluate a single game state.
     */
    std::pair<std::vector<float>, float>
    evaluate_single(const GameState& state);

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool initialized_;
};

} // namespace alphatafl
