#include "inference_engine.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

namespace alphatafl {

InferenceEngine::InferenceEngine(const std::string& model_path, const std::string& device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "AlphaTafl"), initialized_(false), device_("cpu")
{
    try {
        Ort::SessionOptions session_options;

        if (device == "cuda") {
            try {
                OrtCUDAProviderOptions cuda_options;
                std::memset(&cuda_options, 0, sizeof(cuda_options));
                cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[InferenceEngine] Registered CUDA Execution Provider" << std::endl;
            } catch (const std::exception& cuda_err) {
                std::cerr << "[InferenceEngine] CUDA EP registration failed: " << cuda_err.what()
                          << ", falling back to CPU" << std::endl;
            }
        }

        // Set graph optimization level
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Ort::Session requires std::wstring on Windows
        #ifdef _WIN32
        std::wstring w_model_path(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, w_model_path.c_str(), session_options);
        #else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
        #endif

        initialized_ = true;
        device_ = (device == "cuda" && session_) ? "cuda" : "cpu";
        std::cout << "[InferenceEngine] Loaded model from " << model_path
                  << " on " << device_ << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] Error loading model: " << e.what() << std::endl;
        if (device == "cuda") {
            std::cerr << "[InferenceEngine] Retrying model load on CPU..." << std::endl;
            try {
                Ort::SessionOptions cpu_options;
                cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                #ifdef _WIN32
                std::wstring w_model_path(model_path.begin(), model_path.end());
                session_ = std::make_unique<Ort::Session>(env_, w_model_path.c_str(), cpu_options);
                #else
                session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), cpu_options);
                #endif
                initialized_ = true;
                device_ = "cpu";
                std::cout << "[InferenceEngine] Loaded model from " << model_path
                          << " on CPU (fallback)" << std::endl;
            } catch (const std::exception& e2) {
                std::cerr << "[InferenceEngine] CPU fallback loading failed: " << e2.what() << std::endl;
            }
        }
    }
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
InferenceEngine::evaluate(const std::vector<GameState>& states) {
    if (!initialized_ || states.empty()) {
        return {std::vector<std::vector<float>>(states.size(), std::vector<float>(4840, 1.0f / 4840.0f)),
                std::vector<float>(states.size(), 0.0f)};
    }

    const size_t n = states.size();

    // 1. Batch states to flat vector using GameState::batch_to_tensor
    auto flat = GameState::batch_to_tensor(states);  // vector<float>, size = n * 14 * 11 * 11

    // 2. Create Ort input tensor
    std::vector<int64_t> input_shape = { static_cast<int64_t>(n), 14, 11, 11 };
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(flat.data()),
        flat.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 3. Define input and output names
    const char* const input_names[] = {"input"};
    const char* const output_names[] = {"policy", "value"};

    // 4. Run session
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            2
        );
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] session_->Run failed: " << e.what() << std::endl;
        return {std::vector<std::vector<float>>(n, std::vector<float>(4840, 1.0f / 4840.0f)),
                std::vector<float>(n, 0.0f)};
    }

    if (output_tensors.size() < 2) {
        std::cerr << "[InferenceEngine] session_->Run did not return 2 outputs" << std::endl;
        return {std::vector<std::vector<float>>(n, std::vector<float>(4840, 1.0f / 4840.0f)),
                std::vector<float>(n, 0.0f)};
    }

    // 5. Get output data pointers
    float* policy_data = output_tensors[0].GetTensorMutableData<float>();
    float* value_data = output_tensors[1].GetTensorMutableData<float>();

    // 6. Post-processing: masked softmax on CPU
    std::vector<std::vector<float>> policies(n, std::vector<float>(4840, 0.0f));
    std::vector<float> vals(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        auto mask_vec = states[i].get_legal_moves_mask();

        // Apply mask and find max for numerical stability
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < 4840; ++j) {
            if (mask_vec[j] > 0.5f) {
                float logit = policy_data[i * 4840 + j];
                if (logit > max_logit) max_logit = logit;
            }
        }

        // Compute softmax sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < 4840; ++j) {
            if (mask_vec[j] > 0.5f) {
                sum_exp += std::exp(policy_data[i * 4840 + j] - max_logit);
            }
        }

        // Fill policies
        if (sum_exp > 0.0f) {
            for (size_t j = 0; j < 4840; ++j) {
                if (mask_vec[j] > 0.5f) {
                    policies[i][j] = std::exp(policy_data[i * 4840 + j] - max_logit) / sum_exp;
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

        vals[i] = value_data[i];
    }

    return {policies, vals};
}

std::pair<std::vector<float>, float>
InferenceEngine::evaluate_single(const GameState& state) {
    auto batch_result = evaluate({state});
    return {batch_result.first[0], batch_result.second[0]};
}

} // namespace alphatafl
