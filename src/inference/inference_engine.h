#pragma once

#include <string>
#include <vector>
#include <memory>
#include "onnxruntime_cxx_api.h"
#include "game_state.h"

namespace alphatafl {

/**
 * Native C++ inference engine using stand-alone ONNX Runtime.
 * Eliminates PyTorch/LibTorch dependency and runs CUDA or CPU inference.
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
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    bool initialized_;
    std::string device_;
};

} // namespace alphatafl
