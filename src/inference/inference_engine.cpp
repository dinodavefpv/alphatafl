#include "inference_engine.h"
#include <torch/torch.h>
#include <iostream>
#include <cmath>

namespace alphatafl {

InferenceEngine::InferenceEngine(const std::string& model_path, const std::string& device)
    : device_(torch::kCPU), initialized_(false)
{
    try {
        bool cuda_available = torch::cuda::is_available();
        std::cout << "[InferenceEngine] torch::cuda::is_available() = " << cuda_available << std::endl;

        if (device == "cuda") {
            // Force CUDA even if is_available() returns false (pip libtorch quirk)
            try {
                device_ = torch::Device(torch::kCUDA);
                // Test if CUDA actually works
                auto test_tensor = torch::zeros({1}, torch::TensorOptions().device(device_));
                std::cout << "[InferenceEngine] CUDA test tensor created successfully" << std::endl;
            } catch (const c10::Error& cuda_err) {
                std::cerr << "[InferenceEngine] CUDA initialization failed: " << cuda_err.what()
                          << ", falling back to CPU" << std::endl;
                device_ = torch::Device(torch::kCPU);
            }
        } else {
            device_ = torch::Device(torch::kCPU);
        }

        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        initialized_ = true;

        std::cout << "[InferenceEngine] Loaded model from " << model_path
                  << " on " << (device_.is_cuda() ? "cuda" : "cpu") << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[InferenceEngine] Error loading model: " << e.what() << std::endl;
    }
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
InferenceEngine::evaluate(const std::vector<GameState>& states) {
    if (!initialized_ || states.empty()) {
        return {std::vector<std::vector<float>>(states.size(), std::vector<float>(4840, 1.0f / 4840.0f)),
                std::vector<float>(states.size(), 0.0f)};
    }

    const size_t n = states.size();

    // 1. Convert states to tensor using existing batch_to_tensor
    auto flat = GameState::batch_to_tensor(states);  // vector<float>, size = n * 14 * 11 * 11
    // from_blob uses host memory; create CPU tensor then move to device
    auto input = torch::from_blob(flat.data(), {static_cast<long>(n), 14, 11, 11},
                                  torch::TensorOptions().dtype(torch::kFloat32)).to(device_);

    // 2. Run inference
    std::vector<torch::jit::IValue> inputs = {input};
    auto output = model_.forward(inputs).toTuple();
    auto policy_logits = output->elements()[0].toTensor().to(torch::kCPU);  // [N, 4840]
    auto values = output->elements()[1].toTensor().to(torch::kCPU);         // [N, 1]

    // 3. Apply legal masks, softmax, and extract results
    std::vector<std::vector<float>> policies(n, std::vector<float>(4840, 0.0f));
    std::vector<float> vals(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        // Get legal mask
        auto mask_vec = states[i].get_legal_moves_mask();

        // Apply mask and find max for numerical stability
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < 4840; ++j) {
            if (mask_vec[j] > 0.5f) {
                float logit = policy_logits[i][j].item<float>();
                if (logit > max_logit) max_logit = logit;
            }
        }

        // Compute softmax sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < 4840; ++j) {
            if (mask_vec[j] > 0.5f) {
                sum_exp += std::exp(policy_logits[i][j].item<float>() - max_logit);
            }
        }

        // Fill policies
        if (sum_exp > 0.0f) {
            for (size_t j = 0; j < 4840; ++j) {
                if (mask_vec[j] > 0.5f) {
                    policies[i][j] = std::exp(policy_logits[i][j].item<float>() - max_logit) / sum_exp;
                }
            }
        } else {
            // Uniform fallback
            int legal_count = 0;
            for (size_t j = 0; j < 4840; ++j) {
                if (mask_vec[j] > 0.5f) legal_count++;
            }
            if (legal_count > 0) {
                float uniform = 1.0f / legal_count;
                for (size_t j = 0; j < 4840; ++j) {
                    if (mask_vec[j] > 0.5f) policies[i][j] = uniform;
                }
            }
        }

        vals[i] = values[i][0].item<float>();
    }

    return {policies, vals};
}

std::pair<std::vector<float>, float>
InferenceEngine::evaluate_single(const GameState& state) {
    auto batch_result = evaluate({state});
    return {batch_result.first[0], batch_result.second[0]};
}

} // namespace alphatafl
