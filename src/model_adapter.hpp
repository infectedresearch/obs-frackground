/*
 * SPDX-FileCopyrightText: 2026 obs-frackground contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "onnx_backend.hpp"

#include <string>

enum class QualityMode {
    Performance,
    Balanced,
    Quality,
};

struct ModelRuntimeConfig {
    QualityMode quality_mode = QualityMode::Balanced;
    int inference_width = 512;
    int inference_height = 512;
    float temporal_smoothing = 0.13f;
};

class ModelAdapter {
public:
    virtual ~ModelAdapter() = default;

    [[nodiscard]] virtual const char *name() const noexcept = 0;
    [[nodiscard]] virtual std::string validate(const OnnxModelMetadata &metadata,
                                               const ModelRuntimeConfig &config) const = 0;
};

class RvmModelAdapter final : public ModelAdapter {
public:
    [[nodiscard]] const char *name() const noexcept override { return "Robust Video Matting"; }
    [[nodiscard]] std::string validate(const OnnxModelMetadata &metadata,
                                       const ModelRuntimeConfig &config) const override;
};
