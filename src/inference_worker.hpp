/*
 * SPDX-FileCopyrightText: 2026 obs-frackground contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "onnx_backend.hpp"
#include "model_adapter.hpp"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class InferenceWorker {
public:
    InferenceWorker();
    ~InferenceWorker();

    InferenceWorker(const InferenceWorker &) = delete;
    InferenceWorker &operator=(const InferenceWorker &) = delete;

    void start();
    void stop();
    void configure(std::string model_path, bool prefer_cuda, ModelRuntimeConfig config);
    bool submit_frame(std::vector<float> rgb_chw, int width, int height, float downsample_ratio);
    bool latest_matte(RvmFrameResult &result) const;
    uint64_t latest_matte_sequence() const;

    [[nodiscard]] std::string status() const;
    [[nodiscard]] bool model_loaded() const;
    [[nodiscard]] bool cuda_available() const noexcept { return backend_.cuda_available(); }
    [[nodiscard]] bool migraphx_available() const noexcept { return backend_.migraphx_available(); }
    [[nodiscard]] const std::vector<std::string> &providers() const noexcept { return backend_.providers(); }

private:
    void run();

    OnnxBackend backend_;
    RvmModelAdapter adapter_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    std::string model_path_;
    std::string status_;
    ModelRuntimeConfig config_;
    std::vector<float> pending_rgb_chw_;
    RvmFrameResult latest_matte_;
    uint64_t latest_matte_sequence_ = 0;
    int pending_width_ = 0;
    int pending_height_ = 0;
    float pending_downsample_ratio_ = 1.0f;
    bool prefer_cuda_ = true;
    bool loaded_ = false;
    bool running_ = false;
    bool reconfigure_ = false;
    bool frame_pending_ = false;
    bool logged_inference_ok_ = false;
    bool logged_inference_error_ = false;
    int inference_ok_log_count_ = 0;
};
