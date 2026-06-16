/*
 * SPDX-FileCopyrightText: 2026 obs-frackground contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <obs-module.h>

#include "inference_worker.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-frackground", "en-US")

namespace {

constexpr const char *kFilterId = "obs_frackground_filter";
constexpr const char *kEffectFile = "frackground.effect";

struct FrackgroundFilter {
    obs_source_t *source = nullptr;
    gs_effect_t *effect = nullptr;
    gs_texrender_t *source_render = nullptr;
    gs_stagesurf_t *stage_surfaces[2] = {nullptr, nullptr};
    gs_texture_t *matte_texture = nullptr;
    std::unique_ptr<InferenceWorker> inference_worker;
    std::string model_path;
    ModelRuntimeConfig model_config;
    std::vector<uint8_t> matte_upload;
    uint32_t matte_width = 0;
    uint32_t matte_height = 0;
    uint32_t stage_width = 0;
    uint32_t stage_height = 0;
    uint64_t last_uploaded_matte_sequence = 0;
    int stage_index = 0;
    int raw_frame_grace = 0;
    bool stage_valid[2] = {false, false};
    float test_matte_strength = 0.0f;
    float foreground_protection = 0.75f;
    float edge_softness = 0.35f;
    float temporal_smoothing = 0.13f;
    bool prefer_cuda = true;
    bool debug_view = false;
    bool logged_model_ready = false;
    bool logged_first_submit = false;
    bool logged_filter_video_waiting = false;
    bool logged_first_matte = false;
    bool logged_stage_wait = false;
    bool logged_stage_map_failed = false;
    bool logged_stage_ready = false;
    bool logged_no_source_frame = false;
    bool logged_unsupported_source_frame = false;
    bool logged_no_latest_matte = false;
    bool logged_matte_texture_failed = false;
    bool logged_filter_video_ready = false;
    bool logged_submit_dropped = false;
    int matte_upload_log_count = 0;
    int submit_log_count = 0;
    int filter_video_log_count = 0;
};

static const char *filter_name(void *)
{
    return obs_module_text("FrackgroundFilter");
}

static void load_effect(FrackgroundFilter *filter)
{
    char *effect_path = obs_module_file(kEffectFile);
    if (!effect_path) {
        blog(LOG_ERROR, "[obs-frackground] Failed to resolve %s", kEffectFile);
        return;
    }

    char *error_string = nullptr;
    filter->effect = gs_effect_create_from_file(effect_path, &error_string);
    if (!filter->effect) {
        blog(LOG_ERROR, "[obs-frackground] Failed to load effect %s: %s", effect_path,
             error_string ? error_string : "unknown error");
    }

    bfree(error_string);
    bfree(effect_path);
}

static void filter_update(void *data, obs_data_t *settings)
{
    auto *filter = static_cast<FrackgroundFilter *>(data);
    filter->test_matte_strength = static_cast<float>(obs_data_get_double(settings, "test_matte_strength"));
    filter->foreground_protection = static_cast<float>(obs_data_get_double(settings, "foreground_protection"));
    filter->edge_softness = static_cast<float>(obs_data_get_double(settings, "edge_softness"));
    filter->temporal_smoothing = static_cast<float>(obs_data_get_double(settings, "temporal_smoothing"));
    filter->model_path = obs_data_get_string(settings, "model_path");
    const std::string quality_mode = obs_data_get_string(settings, "quality_mode");
    if (quality_mode == "performance")
        filter->model_config.quality_mode = QualityMode::Performance;
    else if (quality_mode == "quality")
        filter->model_config.quality_mode = QualityMode::Quality;
    else
        filter->model_config.quality_mode = QualityMode::Balanced;
    filter->model_config.inference_width = static_cast<int>(obs_data_get_int(settings, "inference_width"));
    filter->model_config.inference_height = static_cast<int>(obs_data_get_int(settings, "inference_height"));
    filter->model_config.temporal_smoothing = filter->temporal_smoothing;
    filter->prefer_cuda = obs_data_get_bool(settings, "prefer_cuda");
    filter->debug_view = obs_data_get_bool(settings, "debug_view");
    filter->logged_model_ready = false;
    filter->logged_first_submit = false;
    filter->logged_filter_video_waiting = false;
    filter->logged_first_matte = false;
    filter->logged_stage_wait = false;
    filter->logged_stage_map_failed = false;
    filter->logged_stage_ready = false;
    filter->logged_no_source_frame = false;
    filter->logged_unsupported_source_frame = false;
    filter->logged_no_latest_matte = false;
    filter->logged_matte_texture_failed = false;
    filter->logged_filter_video_ready = false;
    filter->logged_submit_dropped = false;
    filter->last_uploaded_matte_sequence = 0;
    filter->matte_upload_log_count = 0;
    filter->submit_log_count = 0;
    filter->filter_video_log_count = 0;

    filter->test_matte_strength = std::clamp(filter->test_matte_strength, 0.0f, 1.0f);
    filter->foreground_protection = std::clamp(filter->foreground_protection, 0.0f, 1.0f);
    filter->edge_softness = std::clamp(filter->edge_softness, 0.0f, 1.0f);
    filter->temporal_smoothing = std::clamp(filter->temporal_smoothing, 0.0f, 1.0f);
    filter->model_config.inference_width = std::clamp(filter->model_config.inference_width, 256, 1920);
    filter->model_config.inference_height = std::clamp(filter->model_config.inference_height, 256, 1080);

    if (filter->inference_worker)
        filter->inference_worker->configure(filter->model_path, filter->prefer_cuda, filter->model_config);

    if (filter->inference_worker)
        obs_data_set_string(settings, "runtime_status", filter->inference_worker->status().c_str());
}

static bool frame_format_supported(enum video_format format)
{
    return format == VIDEO_FORMAT_RGBA || format == VIDEO_FORMAT_BGRA || format == VIDEO_FORMAT_BGRX ||
           format == VIDEO_FORMAT_NV12 || format == VIDEO_FORMAT_I420 || format == VIDEO_FORMAT_I40A ||
           format == VIDEO_FORMAT_I422 || format == VIDEO_FORMAT_YUY2 || format == VIDEO_FORMAT_UYVY ||
           format == VIDEO_FORMAT_YVYU;
}

static bool frame_planes_available(const obs_source_frame *frame)
{
    if (!frame || !frame->data[0] || frame->linesize[0] == 0)
        return false;

    switch (frame->format) {
    case VIDEO_FORMAT_NV12:
        return frame->data[1] && frame->linesize[1] > 0;
    case VIDEO_FORMAT_I420:
    case VIDEO_FORMAT_I40A:
    case VIDEO_FORMAT_I422:
        return frame->data[1] && frame->data[2] && frame->linesize[1] > 0 && frame->linesize[2] > 0;
    default:
        return true;
    }
}

static std::pair<int, int> inference_size_for_dimensions(int source_width, int source_height,
                                                         const ModelRuntimeConfig &config)
{
    const int max_width = std::min(config.inference_width, source_width);
    const int max_height = std::min(config.inference_height, source_height);

    if (source_width <= 0 || source_height <= 0 || max_width <= 0 || max_height <= 0)
        return {0, 0};

    const float scale = std::min(static_cast<float>(max_width) / static_cast<float>(source_width),
                                 static_cast<float>(max_height) / static_cast<float>(source_height));
    int width = std::max(2, static_cast<int>(std::lround(static_cast<float>(source_width) * scale)));
    int height = std::max(2, static_cast<int>(std::lround(static_cast<float>(source_height) * scale)));

    width -= width % 2;
    height -= height % 2;
    return {std::max(width, 2), std::max(height, 2)};
}

static std::pair<int, int> inference_size_for(const obs_source_frame *frame, const ModelRuntimeConfig &config)
{
    return inference_size_for_dimensions(static_cast<int>(frame->width), static_cast<int>(frame->height), config);
}

static float rvm_downsample_ratio_for(QualityMode mode, int width, int height)
{
    const int max_dimension = std::max(width, height);
    if (max_dimension <= 512)
        return 1.0f;

    float target_dimension = 384.0f;
    switch (mode) {
    case QualityMode::Performance:
        target_dimension = 256.0f;
        break;
    case QualityMode::Quality:
        target_dimension = 512.0f;
        break;
    case QualityMode::Balanced:
    default:
        target_dimension = 384.0f;
        break;
    }

    return std::clamp(target_dimension / static_cast<float>(max_dimension), 0.125f, 1.0f);
}

static float clamp_channel(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

static void yuv_to_rgb(float y_value, float u_value, float v_value, bool full_range, float &r, float &g, float &b)
{
    const float y = full_range ? y_value : std::max(0.0f, (y_value * 255.0f - 16.0f) / 219.0f);
    const float u = u_value - 0.5f;
    const float v = v_value - 0.5f;

    r = clamp_channel(y + 1.5748f * v);
    g = clamp_channel(y - 0.1873f * u - 0.4681f * v);
    b = clamp_channel(y + 1.8556f * u);
}

static void sample_packed_rgb(const obs_source_frame *frame, int sx, int sy, float &r, float &g, float &b)
{
    const auto *pixel = frame->data[0] + static_cast<size_t>(sy) * frame->linesize[0] + static_cast<size_t>(sx) * 4;

    if (frame->format == VIDEO_FORMAT_RGBA) {
        r = static_cast<float>(pixel[0]) / 255.0f;
        g = static_cast<float>(pixel[1]) / 255.0f;
        b = static_cast<float>(pixel[2]) / 255.0f;
    } else {
        r = static_cast<float>(pixel[2]) / 255.0f;
        g = static_cast<float>(pixel[1]) / 255.0f;
        b = static_cast<float>(pixel[0]) / 255.0f;
    }
}

static void sample_nv12(const obs_source_frame *frame, int sx, int sy, float &r, float &g, float &b)
{
    const auto *y_plane = frame->data[0];
    const auto *uv_plane = frame->data[1];
    const uint8_t y = y_plane[static_cast<size_t>(sy) * frame->linesize[0] + sx];
    const size_t uv_offset = static_cast<size_t>(sy / 2) * frame->linesize[1] + static_cast<size_t>(sx / 2) * 2;
    const uint8_t u = uv_plane[uv_offset];
    const uint8_t v = uv_plane[uv_offset + 1];
    yuv_to_rgb(static_cast<float>(y) / 255.0f, static_cast<float>(u) / 255.0f, static_cast<float>(v) / 255.0f,
               frame->full_range, r, g, b);
}

static void sample_i420(const obs_source_frame *frame, int sx, int sy, float &r, float &g, float &b)
{
    const uint8_t y = frame->data[0][static_cast<size_t>(sy) * frame->linesize[0] + sx];
    const uint8_t u = frame->data[1][static_cast<size_t>(sy / 2) * frame->linesize[1] + sx / 2];
    const uint8_t v = frame->data[2][static_cast<size_t>(sy / 2) * frame->linesize[2] + sx / 2];
    yuv_to_rgb(static_cast<float>(y) / 255.0f, static_cast<float>(u) / 255.0f, static_cast<float>(v) / 255.0f,
               frame->full_range, r, g, b);
}

static void sample_i422(const obs_source_frame *frame, int sx, int sy, float &r, float &g, float &b)
{
    const uint8_t y = frame->data[0][static_cast<size_t>(sy) * frame->linesize[0] + sx];
    const uint8_t u = frame->data[1][static_cast<size_t>(sy) * frame->linesize[1] + sx / 2];
    const uint8_t v = frame->data[2][static_cast<size_t>(sy) * frame->linesize[2] + sx / 2];
    yuv_to_rgb(static_cast<float>(y) / 255.0f, static_cast<float>(u) / 255.0f, static_cast<float>(v) / 255.0f,
               frame->full_range, r, g, b);
}

static void sample_packed_422(const obs_source_frame *frame, int sx, int sy, float &r, float &g, float &b)
{
    const auto *pair = frame->data[0] + static_cast<size_t>(sy) * frame->linesize[0] + static_cast<size_t>(sx / 2) * 4;
    uint8_t y = 0;
    uint8_t u = 0;
    uint8_t v = 0;

    if (frame->format == VIDEO_FORMAT_YUY2) {
        y = pair[(sx % 2) == 0 ? 0 : 2];
        u = pair[1];
        v = pair[3];
    } else if (frame->format == VIDEO_FORMAT_UYVY) {
        y = pair[(sx % 2) == 0 ? 1 : 3];
        u = pair[0];
        v = pair[2];
    } else {
        y = pair[(sx % 2) == 0 ? 0 : 2];
        v = pair[1];
        u = pair[3];
    }

    yuv_to_rgb(static_cast<float>(y) / 255.0f, static_cast<float>(u) / 255.0f, static_cast<float>(v) / 255.0f,
               frame->full_range, r, g, b);
}

static void sample_frame_rgb(const obs_source_frame *frame, int sx, int sy, float &r, float &g, float &b)
{
    switch (frame->format) {
    case VIDEO_FORMAT_RGBA:
    case VIDEO_FORMAT_BGRA:
    case VIDEO_FORMAT_BGRX:
        sample_packed_rgb(frame, sx, sy, r, g, b);
        break;
    case VIDEO_FORMAT_NV12:
        sample_nv12(frame, sx, sy, r, g, b);
        break;
    case VIDEO_FORMAT_I420:
    case VIDEO_FORMAT_I40A:
        sample_i420(frame, sx, sy, r, g, b);
        break;
    case VIDEO_FORMAT_I422:
        sample_i422(frame, sx, sy, r, g, b);
        break;
    case VIDEO_FORMAT_YUY2:
    case VIDEO_FORMAT_UYVY:
    case VIDEO_FORMAT_YVYU:
        sample_packed_422(frame, sx, sy, r, g, b);
        break;
    default:
        r = g = b = 0.0f;
        break;
    }
}

static std::vector<float> convert_frame_to_rgb_chw(const obs_source_frame *frame, int output_width, int output_height)
{
    std::vector<float> rgb(static_cast<size_t>(3 * output_width * output_height));
    const float scale_x = static_cast<float>(frame->width) / static_cast<float>(output_width);
    const float scale_y = static_cast<float>(frame->height) / static_cast<float>(output_height);
    const size_t plane_size = static_cast<size_t>(output_width * output_height);

    for (int y = 0; y < output_height; ++y) {
        const int sy = std::min(static_cast<int>(std::floor((static_cast<float>(y) + 0.5f) * scale_y)),
                                static_cast<int>(frame->height) - 1);

        for (int x = 0; x < output_width; ++x) {
            const int sx = std::min(static_cast<int>(std::floor((static_cast<float>(x) + 0.5f) * scale_x)),
                                    static_cast<int>(frame->width) - 1);
            const size_t dst = static_cast<size_t>(y * output_width + x);
            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;

            sample_frame_rgb(frame, sx, sy, r, g, b);
            rgb[dst] = r;
            rgb[plane_size + dst] = g;
            rgb[plane_size * 2 + dst] = b;
        }
    }

    return rgb;
}

static std::vector<float> convert_rgba_texture_data_to_rgb_chw(const uint8_t *data, uint32_t linesize,
                                                               int source_width, int source_height, int output_width,
                                                               int output_height)
{
    std::vector<float> rgb(static_cast<size_t>(3 * output_width * output_height));
    const float scale_x = static_cast<float>(source_width) / static_cast<float>(output_width);
    const float scale_y = static_cast<float>(source_height) / static_cast<float>(output_height);
    const size_t plane_size = static_cast<size_t>(output_width * output_height);

    for (int y = 0; y < output_height; ++y) {
        const int sy = std::min(static_cast<int>(std::floor((static_cast<float>(y) + 0.5f) * scale_y)), source_height - 1);
        const auto *row = data + static_cast<size_t>(sy) * linesize;

        for (int x = 0; x < output_width; ++x) {
            const int sx = std::min(static_cast<int>(std::floor((static_cast<float>(x) + 0.5f) * scale_x)),
                                    source_width - 1);
            const auto *pixel = row + static_cast<size_t>(sx) * 4;
            const size_t dst = static_cast<size_t>(y * output_width + x);
            rgb[dst] = static_cast<float>(pixel[0]) / 255.0f;
            rgb[plane_size + dst] = static_cast<float>(pixel[1]) / 255.0f;
            rgb[plane_size * 2 + dst] = static_cast<float>(pixel[2]) / 255.0f;
        }
    }

    return rgb;
}

static obs_source_frame *filter_video(void *data, obs_source_frame *frame)
{
    auto *filter = static_cast<FrackgroundFilter *>(data);
    if (!filter || !frame || !filter->inference_worker)
        return frame;

    if (!filter->inference_worker->model_loaded()) {
        if (!filter->logged_filter_video_waiting) {
            blog(LOG_INFO, "[obs-frackground] filter_video is receiving frames; waiting for model load");
            filter->logged_filter_video_waiting = true;
        }
        return frame;
    }

    if (!filter->logged_filter_video_ready) {
        blog(LOG_INFO, "[obs-frackground] filter_video has model and frame: format=%d %ux%u linesize=[%u,%u,%u]",
             static_cast<int>(frame->format), frame->width, frame->height, frame->linesize[0], frame->linesize[1],
             frame->linesize[2]);
        filter->logged_filter_video_ready = true;
    } else if (++filter->filter_video_log_count % 120 == 0) {
        blog(LOG_INFO, "[obs-frackground] filter_video frame count=%d format=%d", filter->filter_video_log_count,
             static_cast<int>(frame->format));
    }

    if (!frame_format_supported(frame->format) || !frame_planes_available(frame)) {
        if (!filter->logged_unsupported_source_frame) {
            blog(LOG_WARNING,
                 "[obs-frackground] Raw frame format unsupported or incomplete: format=%d planes=%s %ux%u linesize=[%u,%u,%u]",
                 static_cast<int>(frame->format), frame_planes_available(frame) ? "yes" : "no", frame->width, frame->height,
                 frame->linesize[0], frame->linesize[1], frame->linesize[2]);
            filter->logged_unsupported_source_frame = true;
        }
        return frame;
    }

    const auto [output_width, output_height] = inference_size_for(frame, filter->model_config);
    if (output_width <= 0 || output_height <= 0)
        return frame;

    auto rgb = convert_frame_to_rgb_chw(frame, output_width, output_height);
    const float downsample_ratio = rvm_downsample_ratio_for(filter->model_config.quality_mode, output_width, output_height);
    if (filter->inference_worker->submit_frame(std::move(rgb), output_width, output_height, downsample_ratio)) {
        if (!filter->logged_first_submit) {
            blog(LOG_INFO, "[obs-frackground] Submitted raw frame for inference: %ux%u -> %dx%d", frame->width,
                 frame->height, output_width, output_height);
            filter->logged_first_submit = true;
        }
        filter->raw_frame_grace = 120;
    } else if (!filter->logged_submit_dropped) {
        blog(LOG_WARNING, "[obs-frackground] Raw frame submission dropped: model_loaded=%s size=%zu expected=%zu",
             filter->inference_worker->model_loaded() ? "yes" : "no", rgb.size(),
             static_cast<size_t>(3 * output_width * output_height));
        filter->logged_submit_dropped = true;
    }

    return frame;
}

static void submit_source_frame_for_inference(FrackgroundFilter *filter, obs_source_t *target)
{
    if (!filter || !target || !filter->inference_worker || !filter->inference_worker->model_loaded())
        return;

    obs_source_frame *frame = obs_source_get_frame(target);
    if (!frame) {
        if (!filter->logged_no_source_frame) {
            blog(LOG_INFO, "[obs-frackground] Target source did not provide an async frame for direct inference");
            filter->logged_no_source_frame = true;
        }
        return;
    }

    if (frame_format_supported(frame->format) && frame_planes_available(frame)) {
        const auto [output_width, output_height] = inference_size_for(frame, filter->model_config);
        if (output_width > 0 && output_height > 0) {
            auto rgb = convert_frame_to_rgb_chw(frame, output_width, output_height);
            const float downsample_ratio = rvm_downsample_ratio_for(filter->model_config.quality_mode, output_width, output_height);
            if (filter->inference_worker->submit_frame(std::move(rgb), output_width, output_height, downsample_ratio)) {
                if (!filter->logged_first_submit) {
                    blog(LOG_INFO, "[obs-frackground] Submitted source frame for inference: %ux%u -> %dx%d", frame->width,
                         frame->height, output_width, output_height);
                    filter->logged_first_submit = true;
                }
                filter->raw_frame_grace = 120;
            }
        }
    } else if (!filter->logged_unsupported_source_frame) {
        blog(LOG_WARNING, "[obs-frackground] Target source frame format unsupported or incomplete: format=%d planes=%s %ux%u",
             static_cast<int>(frame->format), frame_planes_available(frame) ? "yes" : "no", frame->width, frame->height);
        filter->logged_unsupported_source_frame = true;
    }

    obs_source_release_frame(target, frame);
}

static void update_matte_texture(FrackgroundFilter *filter)
{
    RvmFrameResult matte;
    if (!filter->inference_worker || !filter->inference_worker->latest_matte(matte)) {
        if (filter->inference_worker && filter->inference_worker->model_loaded() && !filter->logged_no_latest_matte) {
            blog(LOG_INFO, "[obs-frackground] No latest matte available on render thread yet");
            filter->logged_no_latest_matte = true;
        }
        return;
    }

    const uint64_t matte_sequence = filter->inference_worker->latest_matte_sequence();
    if (matte_sequence != 0 && matte_sequence == filter->last_uploaded_matte_sequence)
        return;

    if (matte.width <= 0 || matte.height <= 0 || matte.alpha.empty())
        return;

    const uint32_t width = static_cast<uint32_t>(matte.width);
    const uint32_t height = static_cast<uint32_t>(matte.height);
    filter->matte_upload.resize(static_cast<size_t>(width * height));

    for (size_t i = 0; i < filter->matte_upload.size(); ++i) {
        const float protected_alpha = std::clamp(matte.alpha[i] + filter->foreground_protection * 0.05f, 0.0f, 1.0f);
        filter->matte_upload[i] = static_cast<uint8_t>(std::lround(protected_alpha * 255.0f));
    }

    if (!filter->matte_texture || filter->matte_width != width || filter->matte_height != height) {
        if (filter->matte_texture)
            gs_texture_destroy(filter->matte_texture);

        const uint8_t *data = filter->matte_upload.data();
        filter->matte_texture = gs_texture_create(width, height, GS_R8, 1, &data, GS_DYNAMIC);
        if (!filter->matte_texture && !filter->logged_matte_texture_failed) {
            blog(LOG_ERROR, "[obs-frackground] Failed to create matte texture: %ux%u", width, height);
            filter->logged_matte_texture_failed = true;
        }
        filter->matte_width = width;
        filter->matte_height = height;
    } else {
        gs_texture_set_image(filter->matte_texture, filter->matte_upload.data(), width, false);
    }

    filter->last_uploaded_matte_sequence = matte_sequence;

    if (!filter->logged_first_matte) {
        blog(LOG_INFO, "[obs-frackground] Uploaded first matte texture: %ux%u seq=%llu", width, height,
             static_cast<unsigned long long>(matte_sequence));
        filter->logged_first_matte = true;
    } else if (++filter->matte_upload_log_count % 10 == 0) {
        blog(LOG_INFO, "[obs-frackground] Uploaded matte texture seq=%llu", static_cast<unsigned long long>(matte_sequence));
    }
}

static void reset_stage_surfaces(FrackgroundFilter *filter)
{
    for (auto *&surface : filter->stage_surfaces) {
        if (surface) {
            gs_stagesurface_destroy(surface);
            surface = nullptr;
        }
    }

    filter->stage_width = 0;
    filter->stage_height = 0;
    filter->stage_index = 0;
    filter->stage_valid[0] = false;
    filter->stage_valid[1] = false;
}

static bool ensure_stage_surfaces(FrackgroundFilter *filter, uint32_t width, uint32_t height)
{
    if (filter->stage_surfaces[0] && filter->stage_surfaces[1] && filter->stage_width == width &&
        filter->stage_height == height)
        return true;

    reset_stage_surfaces(filter);
    filter->stage_surfaces[0] = gs_stagesurface_create(width, height, GS_RGBA);
    filter->stage_surfaces[1] = gs_stagesurface_create(width, height, GS_RGBA);
    filter->stage_width = width;
    filter->stage_height = height;

    if (!filter->stage_surfaces[0] || !filter->stage_surfaces[1]) {
        reset_stage_surfaces(filter);
        return false;
    }

    return true;
}

static void submit_staged_texture_frame(FrackgroundFilter *filter, uint32_t width, uint32_t height)
{
    if (!filter->inference_worker || !filter->inference_worker->model_loaded())
        return;

    const int read_index = filter->stage_index;
    if (!filter->stage_valid[read_index]) {
        if (!filter->logged_stage_wait) {
            blog(LOG_INFO, "[obs-frackground] Waiting for staged texture readback");
            filter->logged_stage_wait = true;
        }
        return;
    }

    uint8_t *data = nullptr;
    uint32_t linesize = 0;
    if (!gs_stagesurface_map(filter->stage_surfaces[read_index], &data, &linesize)) {
        if (!filter->logged_stage_map_failed) {
            blog(LOG_WARNING, "[obs-frackground] Failed to map staged texture readback");
            filter->logged_stage_map_failed = true;
        }
        return;
    }

    if (!filter->logged_stage_ready) {
        blog(LOG_INFO, "[obs-frackground] Mapped staged texture readback: %ux%u linesize=%u", width, height, linesize);
        filter->logged_stage_ready = true;
    }

    const auto [output_width, output_height] = inference_size_for_dimensions(static_cast<int>(width), static_cast<int>(height),
                                                                             filter->model_config);
    if (output_width > 0 && output_height > 0) {
        auto rgb = convert_rgba_texture_data_to_rgb_chw(data, linesize, static_cast<int>(width), static_cast<int>(height),
                                                        output_width, output_height);
        const float downsample_ratio = rvm_downsample_ratio_for(filter->model_config.quality_mode, output_width, output_height);
        if (filter->inference_worker->submit_frame(std::move(rgb), output_width, output_height, downsample_ratio) &&
            !filter->logged_first_submit) {
            blog(LOG_INFO, "[obs-frackground] Submitted rendered texture for inference: %ux%u -> %dx%d", width, height,
                 output_width, output_height);
            filter->logged_first_submit = true;
        } else if (++filter->submit_log_count % 60 == 0) {
            blog(LOG_INFO, "[obs-frackground] Submitted rendered texture frame count=%d", filter->submit_log_count);
        }
    }

    gs_stagesurface_unmap(filter->stage_surfaces[read_index]);
}

static gs_texture_t *render_target_to_texture(FrackgroundFilter *filter, obs_source_t *target, uint32_t width,
                                              uint32_t height, bool submit_for_inference)
{
    if (!filter->source_render)
        filter->source_render = gs_texrender_create(GS_RGBA, GS_ZS_NONE);

    if (!filter->source_render || !gs_texrender_begin(filter->source_render, width, height))
        return nullptr;

    const vec4 clear_color = {0.0f, 0.0f, 0.0f, 0.0f};
    gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
    gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
    obs_source_video_render(target);
    gs_texrender_end(filter->source_render);

    gs_texture_t *texture = gs_texrender_get_texture(filter->source_render);
    if (!texture)
        return nullptr;

    if (submit_for_inference && ensure_stage_surfaces(filter, width, height)) {
        const int write_index = filter->stage_index;
        gs_stage_texture(filter->stage_surfaces[write_index], texture);
        filter->stage_valid[write_index] = true;

        filter->stage_index = 1 - write_index;
        submit_staged_texture_frame(filter, width, height);
    }

    return texture;
}

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
    auto filter = std::make_unique<FrackgroundFilter>();
    filter->source = source;
    filter->inference_worker = std::make_unique<InferenceWorker>();

    std::ostringstream provider_list;
    const auto &providers = filter->inference_worker->providers();
    for (size_t i = 0; i < providers.size(); ++i) {
        if (i > 0)
            provider_list << ", ";
        provider_list << providers[i];
    }
    blog(LOG_INFO, "[obs-frackground] ONNX Runtime providers: %s", provider_list.str().c_str());
    blog(filter->inference_worker->cuda_available() ? LOG_INFO : LOG_WARNING,
         "[obs-frackground] CUDA execution provider %s",
         filter->inference_worker->cuda_available() ? "available" : "not available");
    blog(filter->inference_worker->migraphx_available() ? LOG_INFO : LOG_WARNING,
         "[obs-frackground] MIGraphX execution provider %s",
         filter->inference_worker->migraphx_available() ? "available" : "not available");
    filter->inference_worker->start();

    obs_enter_graphics();
    load_effect(filter.get());
    obs_leave_graphics();

    filter_update(filter.get(), settings);
    blog(LOG_INFO, "[obs-frackground] Filter created");
    return filter.release();
}

static void filter_destroy(void *data)
{
    auto *filter = static_cast<FrackgroundFilter *>(data);
    if (!filter)
        return;

    obs_enter_graphics();
    if (filter->effect)
        gs_effect_destroy(filter->effect);
    if (filter->source_render)
        gs_texrender_destroy(filter->source_render);
    reset_stage_surfaces(filter);
    if (filter->matte_texture)
        gs_texture_destroy(filter->matte_texture);
    obs_leave_graphics();

    delete filter;
}

static void filter_defaults(obs_data_t *settings)
{
    char *default_model_path = obs_module_file("rvm_mobilenetv3_fp16.onnx");

    obs_data_set_default_double(settings, "test_matte_strength", 0.24);
    obs_data_set_default_double(settings, "foreground_protection", 0.0);
    obs_data_set_default_double(settings, "edge_softness", 0.35);
    obs_data_set_default_double(settings, "temporal_smoothing", 0.13);
    obs_data_set_default_string(settings, "model_path", default_model_path ? default_model_path : "");
    obs_data_set_default_string(settings, "quality_mode", "balanced");
    obs_data_set_default_int(settings, "inference_width", 512);
    obs_data_set_default_int(settings, "inference_height", 512);
    obs_data_set_default_bool(settings, "prefer_cuda", true);
    obs_data_set_default_bool(settings, "debug_view", false);
    obs_data_set_default_string(settings, "runtime_status", "No model selected");

    bfree(default_model_path);
}

static obs_properties_t *filter_properties(void *)
{
    obs_properties_t *props = obs_properties_create();

    obs_properties_add_float_slider(props, "test_matte_strength", obs_module_text("TestMatte"), 0.0, 1.0,
                                    0.01);
    obs_properties_add_float_slider(props, "foreground_protection", obs_module_text("ForegroundProtection"),
                                    0.0, 1.0, 0.01);
    obs_properties_add_float_slider(props, "edge_softness", obs_module_text("EdgeSoftness"), 0.0, 1.0,
                                    0.01);
    obs_properties_add_float_slider(props, "temporal_smoothing", obs_module_text("TemporalSmoothing"), 0.0,
                                    1.0, 0.01);
    obs_properties_add_path(props, "model_path", obs_module_text("ModelPath"), OBS_PATH_FILE,
                            "ONNX models (*.onnx)", nullptr);
    obs_property_t *quality = obs_properties_add_list(props, "quality_mode", obs_module_text("QualityMode"),
                                                      OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(quality, obs_module_text("QualityPerformance"), "performance");
    obs_property_list_add_string(quality, obs_module_text("QualityBalanced"), "balanced");
    obs_property_list_add_string(quality, obs_module_text("QualityQuality"), "quality");
    obs_properties_add_int(props, "inference_width", obs_module_text("InferenceWidth"), 256, 1920, 32);
    obs_properties_add_int(props, "inference_height", obs_module_text("InferenceHeight"), 256, 1080, 32);
    obs_properties_add_bool(props, "prefer_cuda", obs_module_text("PreferCuda"));
    obs_properties_add_text(props, "runtime_status", obs_module_text("Status"), OBS_TEXT_INFO);
    obs_properties_add_bool(props, "debug_view", obs_module_text("DebugView"));

    return props;
}

static void filter_render(void *data, gs_effect_t *)
{
    auto *filter = static_cast<FrackgroundFilter *>(data);
    obs_source_t *target = obs_filter_get_target(filter->source);
    if (!target) {
        obs_source_skip_video_filter(filter->source);
        return;
    }

    if (!filter->effect) {
        obs_source_skip_video_filter(filter->source);
        return;
    }

    const uint32_t width = obs_source_get_base_width(target);
    const uint32_t height = obs_source_get_base_height(target);
    if (!width || !height) {
        obs_source_skip_video_filter(filter->source);
        return;
    }

    update_matte_texture(filter);

    if (!obs_source_process_filter_begin(filter->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING)) {
        obs_source_skip_video_filter(filter->source);
        return;
    }

    gs_eparam_t *test_matte = gs_effect_get_param_by_name(filter->effect, "test_matte_strength");
    if (test_matte)
        gs_effect_set_float(test_matte, filter->test_matte_strength);

    gs_eparam_t *edge_softness = gs_effect_get_param_by_name(filter->effect, "edge_softness");
    if (edge_softness)
        gs_effect_set_float(edge_softness, filter->edge_softness);

    gs_eparam_t *debug_view = gs_effect_get_param_by_name(filter->effect, "debug_view");
    if (debug_view)
        gs_effect_set_float(debug_view, filter->debug_view ? 1.0f : 0.0f);

    gs_eparam_t *matte_texel = gs_effect_get_param_by_name(filter->effect, "matte_texel");
    if (matte_texel) {
        struct vec2 texel = {0.0f, 0.0f};
        if (filter->matte_width > 0 && filter->matte_height > 0) {
            texel.x = 1.0f / static_cast<float>(filter->matte_width);
            texel.y = 1.0f / static_cast<float>(filter->matte_height);
        }
        gs_effect_set_vec2(matte_texel, &texel);
    }

    gs_eparam_t *use_model_matte = gs_effect_get_param_by_name(filter->effect, "use_model_matte");
    if (use_model_matte)
        gs_effect_set_float(use_model_matte, filter->matte_texture ? 1.0f : 0.0f);

    gs_eparam_t *matte_param = gs_effect_get_param_by_name(filter->effect, "matte");
    if (matte_param && filter->matte_texture)
        gs_effect_set_texture(matte_param, filter->matte_texture);

    obs_source_process_filter_end(filter->source, filter->effect, width, height);
}

} // namespace

bool obs_module_load(void)
{
    obs_source_info info = {};
    info.id = kFilterId;
    info.type = OBS_SOURCE_TYPE_FILTER;
    info.output_flags = OBS_SOURCE_ASYNC_VIDEO | OBS_SOURCE_SRGB;
    info.get_name = filter_name;
    info.create = filter_create;
    info.destroy = filter_destroy;
    info.update = filter_update;
    info.get_defaults = filter_defaults;
    info.get_properties = filter_properties;
    info.filter_video = filter_video;
    info.video_render = filter_render;

    obs_register_source(&info);
    blog(LOG_INFO, "[obs-frackground] Plugin loaded");
    return true;
}
