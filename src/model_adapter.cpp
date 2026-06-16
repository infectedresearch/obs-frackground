/*
 * SPDX-FileCopyrightText: 2026 obs-frackround contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "model_adapter.hpp"

#include <algorithm>
#include <sstream>

namespace {

bool contains_ci(const std::string &value, const std::string &needle)
{
    auto lower_value = value;
    auto lower_needle = needle;
    std::ranges::transform(lower_value, lower_value.begin(), [](unsigned char c) { return std::tolower(c); });
    std::ranges::transform(lower_needle, lower_needle.begin(), [](unsigned char c) { return std::tolower(c); });
    return lower_value.find(lower_needle) != std::string::npos;
}

bool has_rank4_rgb_input(const OnnxModelMetadata &metadata)
{
    return std::ranges::any_of(metadata.inputs, [](const OnnxTensorInfo &tensor) {
        if (tensor.shape.size() != 4)
            return false;

        return tensor.shape[1] == 3 || tensor.shape[1] == -1;
    });
}

bool has_alpha_output(const OnnxModelMetadata &metadata)
{
    return std::ranges::any_of(metadata.outputs, [](const OnnxTensorInfo &tensor) {
        if (contains_ci(tensor.name, "pha") || contains_ci(tensor.name, "alpha") || contains_ci(tensor.name, "matte"))
            return true;

        return tensor.shape.size() == 4 && (tensor.shape[1] == 1 || tensor.shape[1] == -1);
    });
}

int count_recurrent_tensors(const OnnxModelMetadata &metadata)
{
    return static_cast<int>(std::ranges::count_if(metadata.inputs, [](const OnnxTensorInfo &tensor) {
        return contains_ci(tensor.name, "r1") || contains_ci(tensor.name, "r2") || contains_ci(tensor.name, "r3") ||
               contains_ci(tensor.name, "r4");
    }));
}

} // namespace

std::string RvmModelAdapter::validate(const OnnxModelMetadata &metadata, const ModelRuntimeConfig &config) const
{
    std::ostringstream status;

    if (!has_rank4_rgb_input(metadata))
        return "RVM validation failed: expected a rank-4 RGB input tensor";

    if (!has_alpha_output(metadata))
        return "RVM validation failed: expected an alpha/matte output tensor";

    const int recurrent_inputs = count_recurrent_tensors(metadata);
    status << "RVM adapter ready; inference=" << config.inference_width << "x" << config.inference_height;

    if (recurrent_inputs > 0)
        status << "; recurrent inputs=" << recurrent_inputs;
    else
        status << "; recurrent inputs not detected";

    return status.str();
}
