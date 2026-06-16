/*
 * SPDX-FileCopyrightText: 2026 obs-frackround contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "onnx_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <numeric>
#include <sstream>

namespace {

std::string shape_to_string(const std::vector<int64_t> &shape)
{
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
            out << "x";
        out << (shape[i] < 0 ? "?" : std::to_string(shape[i]));
    }
    out << "]";
    return out.str();
}

std::string describe_session(const Ort::Session &session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    std::ostringstream out;

    out << " inputs=";
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        if (i > 0)
            out << ",";
        auto name = session.GetInputNameAllocated(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        out << name.get() << shape_to_string(tensor_info.GetShape());
    }

    out << " outputs=";
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        if (i > 0)
            out << ",";
        auto name = session.GetOutputNameAllocated(i, allocator);
        auto type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        out << name.get() << shape_to_string(tensor_info.GetShape());
    }

    return out.str();
}

OnnxModelMetadata collect_metadata(const Ort::Session &session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    OnnxModelMetadata metadata;

    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        metadata.inputs.push_back({name.get(), tensor_info.GetShape(), static_cast<int>(tensor_info.GetElementType())});
    }

    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        auto type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        metadata.outputs.push_back({name.get(), tensor_info.GetShape(), static_cast<int>(tensor_info.GetElementType())});
    }

    return metadata;
}

const OnnxTensorInfo *find_tensor(const std::vector<OnnxTensorInfo> &tensors, const std::string &name)
{
    const auto it = std::ranges::find_if(tensors, [&name](const OnnxTensorInfo &tensor) { return tensor.name == name; });
    return it == tensors.end() ? nullptr : &*it;
}

} // namespace

class OnnxBackend::Impl {
public:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "obs-frackround"};
    std::unique_ptr<Ort::Session> session;
    std::array<std::vector<float>, 4> rec_fp32;
    std::array<std::vector<Ort::Float16_t>, 4> rec_fp16;
    std::array<std::vector<int64_t>, 4> rec_shapes;
};

OnnxBackend::OnnxBackend()
    : impl_(std::make_unique<Impl>())
{
    providers_ = Ort::GetAvailableProviders();
    cuda_available_ = std::find(providers_.begin(), providers_.end(), "CUDAExecutionProvider") != providers_.end();
    migraphx_available_ = std::find(providers_.begin(), providers_.end(), "MIGraphXExecutionProvider") != providers_.end();
    status_ = cuda_available_ ? "ONNX Runtime ready; CUDA provider available"
                              : (migraphx_available_ ? "ONNX Runtime ready; MIGraphX provider available"
                                                     : "ONNX Runtime ready; GPU provider unavailable");
}

OnnxBackend::~OnnxBackend() = default;

bool OnnxBackend::load_model(const std::string &model_path, bool prefer_cuda)
{
    unload_model();

    if (model_path.empty()) {
        status_ = "No model selected";
        return false;
    }

    if (!std::filesystem::is_regular_file(model_path)) {
        status_ = "Model file does not exist: " + model_path;
        return false;
    }

    try {
        const auto create_options = [this](const std::string &provider) {
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            if (provider == "CUDA") {
                Ort::CUDAProviderOptions cuda_options;
                cuda_options.Update({{"device_id", "0"}, {"cudnn_conv_algo_search", "HEURISTIC"}});
                options.AppendExecutionProvider_CUDA_V2(*cuda_options);
            } else if (provider == "MIGraphX") {
                OrtMIGraphXProviderOptions migraphx_options = {};
                migraphx_options.device_id = 0;
                migraphx_options.migraphx_fp16_enable = 1;
                migraphx_options.migraphx_mem_limit = std::numeric_limits<size_t>::max();
                options.AppendExecutionProvider_MIGraphX(migraphx_options);
            }

            return options;
        };

        std::vector<std::string> provider_attempts;
        if (prefer_cuda) {
            if (cuda_available_) {
                provider_attempts.push_back("CUDA");
            } else {
                status_ = "CUDA requested but CUDAExecutionProvider is unavailable; falling back to CPU";
            }
        }
        provider_attempts.push_back("CPU");

        std::string provider_used;
        std::string provider_error;
        for (const auto &provider : provider_attempts) {
            try {
                Ort::SessionOptions options = create_options(provider == "CPU" ? "" : provider);
                impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), options);
                provider_used = provider;
                break;
            } catch (const std::exception &error) {
                impl_->session.reset();
                provider_error = error.what();
            }
        }

        if (!impl_->session) {
            status_ = "Failed to load model: " + provider_error;
            return false;
        }

        metadata_ = collect_metadata(*impl_->session);
        model_loaded_ = true;
        status_ = "Model loaded with ONNX Runtime " + provider_used;
        if (!provider_error.empty() && provider_used == "CPU")
            status_ += " (GPU provider failed; using CPU)";
        status_ += describe_session(*impl_->session);
        return true;
    } catch (const std::exception &error) {
        unload_model();
        status_ = std::string("Failed to load model: ") + error.what();
        return false;
    }
}

void OnnxBackend::unload_model()
{
    impl_->session.reset();
    metadata_ = {};
    model_loaded_ = false;
}

bool OnnxBackend::run_rvm_smoke_test(int width, int height, float downsample_ratio, std::string &error)
{
    std::vector<float> rgb(static_cast<size_t>(3 * width * height));
    RvmFrameResult result;
    reset_rvm_state();
    return run_rvm_frame(rgb, width, height, downsample_ratio, result, error);
}

void OnnxBackend::reset_rvm_state()
{
    for (auto &state : impl_->rec_fp32)
        state.clear();
    for (auto &state : impl_->rec_fp16)
        state.clear();
    for (auto &shape : impl_->rec_shapes)
        shape = {1, 1, 1, 1};
}

bool OnnxBackend::run_rvm_frame(const std::vector<float> &rgb_chw, int width, int height, float downsample_ratio,
                                RvmFrameResult &result, std::string &error)
{
    error.clear();
    result = {};

    if (!impl_->session || !model_loaded_) {
        error = "No model loaded";
        return false;
    }

    const OnnxTensorInfo *src_info = find_tensor(metadata_.inputs, "src");
    if (!src_info) {
        error = "RVM inference requires input named src";
        return false;
    }

    const bool fp16 = src_info->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    const bool fp32 = src_info->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if (!fp16 && !fp32) {
        error = "RVM inference supports only FP16/FP32 source tensors";
        return false;
    }

    const size_t expected_values = static_cast<size_t>(3 * width * height);
    if (rgb_chw.size() != expected_values) {
        error = "RVM input frame has wrong size";
        return false;
    }

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 4> src_shape{1, 3, height, width};
        const std::array<int64_t, 4> initial_rec_shape{1, 1, 1, 1};
        const std::array<int64_t, 1> ratio_shape{1};
        std::vector<float> downsample{downsample_ratio};

        std::vector<const char *> input_names{"src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"};
        std::vector<const char *> output_names;
        output_names.reserve(metadata_.outputs.size());
        for (const auto &output : metadata_.outputs)
            output_names.push_back(output.name.c_str());

        std::vector<Ort::Value> inputs;
        inputs.reserve(input_names.size());

        std::vector<float> src_fp32;
        std::vector<Ort::Float16_t> src_fp16;
        if (fp16) {
            src_fp16.reserve(rgb_chw.size());
            for (float value : rgb_chw)
                src_fp16.emplace_back(value);

            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(memory_info, src_fp16.data(), src_fp16.size(),
                                                                      src_shape.data(), src_shape.size()));
            for (size_t i = 0; i < impl_->rec_fp16.size(); ++i) {
                if (impl_->rec_fp16[i].empty()) {
                    impl_->rec_fp16[i] = {Ort::Float16_t{0.0f}};
                    impl_->rec_shapes[i] = {initial_rec_shape.begin(), initial_rec_shape.end()};
                }
                inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                    memory_info, impl_->rec_fp16[i].data(), impl_->rec_fp16[i].size(), impl_->rec_shapes[i].data(),
                    impl_->rec_shapes[i].size()));
            }
        } else {
            src_fp32 = rgb_chw;
            inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, src_fp32.data(), src_fp32.size(),
                                                             src_shape.data(), src_shape.size()));
            for (size_t i = 0; i < impl_->rec_fp32.size(); ++i) {
                if (impl_->rec_fp32[i].empty()) {
                    impl_->rec_fp32[i] = {0.0f};
                    impl_->rec_shapes[i] = {initial_rec_shape.begin(), initial_rec_shape.end()};
                }
                inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, impl_->rec_fp32[i].data(),
                                                                 impl_->rec_fp32[i].size(), impl_->rec_shapes[i].data(),
                                                                 impl_->rec_shapes[i].size()));
            }
        }

        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, downsample.data(), downsample.size(),
                                                         ratio_shape.data(), ratio_shape.size()));

        std::vector<Ort::Value> outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                              inputs.data(), inputs.size(), output_names.data(),
                                                              output_names.size());

        if (outputs.size() < 6) {
            error = "RVM inference returned fewer outputs than expected";
            return false;
        }

        auto alpha_info = outputs[1].GetTensorTypeAndShapeInfo();
        const auto alpha_shape = alpha_info.GetShape();
        if (alpha_shape.size() != 4 || alpha_shape[2] <= 0 || alpha_shape[3] <= 0) {
            error = "RVM alpha output has unexpected shape";
            return false;
        }

        result.width = static_cast<int>(alpha_shape[3]);
        result.height = static_cast<int>(alpha_shape[2]);
        const size_t alpha_values = static_cast<size_t>(result.width * result.height);
        result.alpha.resize(alpha_values);

        if (fp16) {
            const auto *alpha = outputs[1].GetTensorData<Ort::Float16_t>();
            for (size_t i = 0; i < alpha_values; ++i)
                result.alpha[i] = static_cast<float>(alpha[i]);

            for (size_t i = 0; i < impl_->rec_fp16.size(); ++i) {
                auto rec_info = outputs[i + 2].GetTensorTypeAndShapeInfo();
                impl_->rec_shapes[i] = rec_info.GetShape();
                const size_t value_count = rec_info.GetElementCount();
                const auto *rec = outputs[i + 2].GetTensorData<Ort::Float16_t>();
                impl_->rec_fp16[i].assign(rec, rec + value_count);
            }
        } else {
            const auto *alpha = outputs[1].GetTensorData<float>();
            std::copy(alpha, alpha + alpha_values, result.alpha.begin());

            for (size_t i = 0; i < impl_->rec_fp32.size(); ++i) {
                auto rec_info = outputs[i + 2].GetTensorTypeAndShapeInfo();
                impl_->rec_shapes[i] = rec_info.GetShape();
                const size_t value_count = rec_info.GetElementCount();
                const auto *rec = outputs[i + 2].GetTensorData<float>();
                impl_->rec_fp32[i].assign(rec, rec + value_count);
            }
        }

        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}
