/*
 * SPDX-FileCopyrightText: 2026 obs-frackground contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "model_adapter.hpp"
#include "onnx_backend.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "usage: frackground-probe <model.onnx> [cpu]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const bool prefer_cuda = argc < 3 || std::string(argv[2]) != "cpu";

    OnnxBackend backend;
    std::cout << backend.status() << '\n';

    if (!backend.load_model(model_path, prefer_cuda)) {
        std::cerr << backend.status() << '\n';
        return 1;
    }

    RvmModelAdapter adapter;
    ModelRuntimeConfig config;
    std::cout << backend.status() << '\n';
    std::cout << adapter.validate(backend.metadata(), config) << '\n';

    std::string error;
    if (!backend.run_rvm_smoke_test(256, 256, 1.0f, error)) {
        std::cerr << "RVM smoke inference failed: " << error << '\n';
        return 1;
    }

    std::cout << "RVM smoke inference passed\n";

    return 0;
}
