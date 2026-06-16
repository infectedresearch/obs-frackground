/*
 * SPDX-FileCopyrightText: 2026 obs-frackround contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

struct OnnxTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    int element_type = 0;
};

struct OnnxModelMetadata {
    std::vector<OnnxTensorInfo> inputs;
    std::vector<OnnxTensorInfo> outputs;
};

struct RvmFrameResult {
    int width = 0;
    int height = 0;
    std::vector<float> alpha;
};

class OnnxBackend {
public:
    OnnxBackend();
    ~OnnxBackend();

    OnnxBackend(const OnnxBackend &) = delete;
    OnnxBackend &operator=(const OnnxBackend &) = delete;

    bool load_model(const std::string &model_path, bool prefer_cuda);
    bool run_rvm_smoke_test(int width, int height, float downsample_ratio, std::string &error);
    bool run_rvm_frame(const std::vector<float> &rgb_chw, int width, int height, float downsample_ratio,
                       RvmFrameResult &result, std::string &error);
    void reset_rvm_state();
    void unload_model();

    [[nodiscard]] bool cuda_available() const noexcept { return cuda_available_; }
    [[nodiscard]] bool migraphx_available() const noexcept { return migraphx_available_; }
    [[nodiscard]] bool model_loaded() const noexcept { return model_loaded_; }
    [[nodiscard]] const std::string &status() const noexcept { return status_; }
    [[nodiscard]] const OnnxModelMetadata &metadata() const noexcept { return metadata_; }
    [[nodiscard]] const std::vector<std::string> &providers() const noexcept { return providers_; }

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
    std::vector<std::string> providers_;
    OnnxModelMetadata metadata_;
    bool cuda_available_ = false;
    bool migraphx_available_ = false;
    bool model_loaded_ = false;
    std::string status_;
};
