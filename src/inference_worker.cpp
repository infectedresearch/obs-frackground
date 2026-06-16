/*
 * SPDX-FileCopyrightText: 2026 obs-frackround contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "inference_worker.hpp"

#include <obs-module.h>

#include <chrono>
#include <algorithm>
#include <utility>

InferenceWorker::InferenceWorker()
    : status_(backend_.status())
{
}

InferenceWorker::~InferenceWorker()
{
    stop();
}

void InferenceWorker::start()
{
    std::lock_guard lock(mutex_);
    if (running_)
        return;

    running_ = true;
    thread_ = std::thread(&InferenceWorker::run, this);
}

void InferenceWorker::stop()
{
    {
        std::lock_guard lock(mutex_);
        if (!running_)
            return;
        running_ = false;
        reconfigure_ = true;
    }

    condition_.notify_one();
    if (thread_.joinable())
        thread_.join();
}

void InferenceWorker::configure(std::string model_path, bool prefer_cuda, ModelRuntimeConfig config)
{
    {
        std::lock_guard lock(mutex_);
        const bool model_config_unchanged =
            model_path == model_path_ && prefer_cuda == prefer_cuda_ &&
            config.inference_width == config_.inference_width && config.inference_height == config_.inference_height &&
            config.quality_mode == config_.quality_mode;

        if (model_config_unchanged && config.temporal_smoothing == config_.temporal_smoothing)
            return;

        if (model_config_unchanged) {
            config_.temporal_smoothing = config.temporal_smoothing;
            blog(LOG_INFO, "[obs-frackround] Temporal smoothing updated: %.2f", config_.temporal_smoothing);
            return;
        }

        model_path_ = std::move(model_path);
        prefer_cuda_ = prefer_cuda;
        config_ = config;
        reconfigure_ = true;
        frame_pending_ = false;
        logged_inference_ok_ = false;
        logged_inference_error_ = false;
        inference_ok_log_count_ = 0;
        status_ = model_path_.empty() ? "No model selected" : "Model load queued";
    }

    condition_.notify_one();
}

bool InferenceWorker::submit_frame(std::vector<float> rgb_chw, int width, int height, float downsample_ratio)
{
    if (width <= 0 || height <= 0 || rgb_chw.size() != static_cast<size_t>(3 * width * height))
        return false;

    {
        std::lock_guard lock(mutex_);
        if (!running_ || !loaded_)
            return false;

        pending_rgb_chw_ = std::move(rgb_chw);
        pending_width_ = width;
        pending_height_ = height;
        pending_downsample_ratio_ = downsample_ratio;
        frame_pending_ = true;
    }

    condition_.notify_one();
    return true;
}

bool InferenceWorker::latest_matte(RvmFrameResult &result) const
{
    std::lock_guard lock(mutex_);
    if (latest_matte_.alpha.empty())
        return false;

    result = latest_matte_;
    return true;
}

uint64_t InferenceWorker::latest_matte_sequence() const
{
    std::lock_guard lock(mutex_);
    return latest_matte_sequence_;
}

std::string InferenceWorker::status() const
{
    std::lock_guard lock(mutex_);
    return status_;
}

bool InferenceWorker::model_loaded() const
{
    std::lock_guard lock(mutex_);
    return loaded_;
}

void InferenceWorker::run()
{
    while (true) {
        std::string model_path;
        ModelRuntimeConfig config;
        std::vector<float> frame;
        int frame_width = 0;
        int frame_height = 0;
        float downsample_ratio = 1.0f;
        bool prefer_cuda = true;
        bool should_reconfigure = false;
        bool should_run_frame = false;

        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] { return reconfigure_ || frame_pending_; });
            if (!running_)
                break;

            if (reconfigure_) {
                model_path = model_path_;
                config = config_;
                prefer_cuda = prefer_cuda_;
                reconfigure_ = false;
                should_reconfigure = true;
                status_ = model_path.empty() ? "No model selected" : "Loading model";
            } else if (frame_pending_) {
                frame = std::move(pending_rgb_chw_);
                frame_width = pending_width_;
                frame_height = pending_height_;
                downsample_ratio = pending_downsample_ratio_;
                frame_pending_ = false;
                should_run_frame = true;
            }
        }

        if (should_reconfigure) {
            bool loaded = false;
            if (model_path.empty()) {
                backend_.unload_model();
            } else if (backend_.load_model(model_path, prefer_cuda)) {
                backend_.reset_rvm_state();
                const std::string adapter_status = adapter_.validate(backend_.metadata(), config);
                loaded = adapter_status.find("failed") == std::string::npos;
                std::lock_guard lock(mutex_);
                loaded_ = loaded;
                status_ = backend_.status() + "; " + adapter_status;
                continue;
            }

            std::lock_guard lock(mutex_);
            loaded_ = loaded;
            latest_matte_ = {};
            latest_matte_sequence_ = 0;
            status_ = backend_.status();
            continue;
        }

        if (should_run_frame) {
            RvmFrameResult result;
            std::string error;
            const auto start = std::chrono::steady_clock::now();
            const bool ok = backend_.run_rvm_frame(frame, frame_width, frame_height, downsample_ratio, result, error);
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::lock_guard lock(mutex_);
            if (ok) {
                const auto [min_alpha, max_alpha] = std::ranges::minmax(result.alpha);
                const float smoothing = std::clamp(config_.temporal_smoothing, 0.0f, 0.95f);
                if (!latest_matte_.alpha.empty() && latest_matte_.width == result.width &&
                    latest_matte_.height == result.height && smoothing > 0.0f) {
                    for (size_t i = 0; i < result.alpha.size(); ++i)
                        result.alpha[i] = latest_matte_.alpha[i] * smoothing + result.alpha[i] * (1.0f - smoothing);
                }
                latest_matte_ = std::move(result);
                ++latest_matte_sequence_;
                status_ = "Inference ok; last frame " + std::to_string(elapsed_ms) + " ms";
                if (!logged_inference_ok_) {
                    blog(LOG_INFO, "[obs-frackround] %s; matte=%dx%d alpha=[%.3f, %.3f]", status_.c_str(),
                         latest_matte_.width, latest_matte_.height, min_alpha, max_alpha);
                    logged_inference_ok_ = true;
                } else if (++inference_ok_log_count_ % 10 == 0) {
                    blog(LOG_INFO, "[obs-frackround] Inference ok seq=%llu; last frame %lld ms",
                         static_cast<unsigned long long>(latest_matte_sequence_), static_cast<long long>(elapsed_ms));
                }
            } else {
                status_ = "Inference failed: " + error;
                if (!logged_inference_error_) {
                    blog(LOG_ERROR, "[obs-frackround] %s", status_.c_str());
                    logged_inference_error_ = true;
                }
            }
            continue;
        }
    }
}
