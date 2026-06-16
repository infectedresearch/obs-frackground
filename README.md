# obs-frackground

`obs-frackground` is a Linux-first OBS Studio filter plugin for high-quality background removal, auto greenscreen, and transparent background output.

It uses ONNX Runtime with an RVM-style video matting model to produce a soft alpha matte instead of a hard binary mask. The current release focuses on OBS Studio on Linux, CUDA acceleration when available, and graceful CPU fallback for testing.

## Demo

![Frackground background removal demo](docs/media/demo.gif)

Frackground tracks head movement while preserving hair and face detail. A higher-quality MP4 version is available at [`docs/media/demo.mp4`](docs/media/demo.mp4).

## Features

- OBS video filter for person/background separation.
- Transparent output for compositing over any OBS scene.
- Auto greenscreen-style background removal without a physical green screen.
- ONNX Runtime backend with CUDA preference and CPU fallback.
- Async worker thread so model loading and inference do not block OBS rendering.
- Raw frame path for common camera/media formats.
- Debug mask preview for tuning.

## Status

This project is alpha-quality but usable for testing. Linux is the primary target. The binary artifact is expected to be compatible only with similar OBS/libobs and ONNX Runtime versions to the build host.

## Requirements

- Linux x86_64.
- OBS Studio with development headers for building.
- ONNX Runtime C/C++ library and headers.
- Optional: ONNX Runtime CUDA execution provider and a working NVIDIA CUDA stack.
- CMake 3.28 or newer.
- C++20 compiler.

On Arch-style systems the runtime dependency currently resolves as `libonnxruntime.so.1`. Other distributions may package ONNX Runtime differently.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

## Download Model

```sh
bash scripts/download-rvm-models.sh
```

This downloads the RobustVideoMatting MobileNetV3 FP16 ONNX model into `models/`. If present, the model is installed beside the plugin and used as the default model path.

## Local Install

```sh
cmake --install build --prefix ~/.local
```

Restart OBS, add the `Frackground Background Removal` filter to a video source, and select or confirm the `Matting model (.onnx)` path.

## Ready-To-Test Defaults

- Model: `rvm_mobilenetv3_fp16.onnx` installed beside the plugin.
- Backend: prefer CUDA, fall back to CPU if CUDA is unavailable.
- Quality mode: `Balanced`.
- Inference size: `512x512`.
- Test matte strength: `0.24`.
- Foreground protection: `0.0`.
- Edge softness: `0.35`.
- Temporal smoothing: `0.13`.
- Debug view: off.

## Supported Frame Paths

The raw frame path supports these OBS video formats:

- `RGBA`
- `BGRA`
- `BGRX`
- `NV12`
- `I420`
- `I40A`
- `I422`
- `YUY2`
- `UYVY`
- `YVYU`

GPU-rendered sources that do not provide raw frames need additional testing and render-path wiring before they are considered supported in release artifacts.

## Packaging

Create a local Linux release artifact with:

```sh
bash scripts/package-linux.sh
```

The artifact is written to `dist/` and uses OBS's expected user install layout.

## Troubleshooting

- If OBS does not show the filter, confirm `obs-frackground.so` is in `~/.local/lib/obs-plugins/` and restart OBS.
- If the filter loads but no matte appears, check the `Runtime status` field and OBS logs for model load or inference errors.
- If CUDA is unavailable, the plugin should fall back to CPU. CPU inference may be too slow for real-time use.
- If the model path is empty, install or download `rvm_mobilenetv3_fp16.onnx` and select it in the filter settings.
- If you upgraded from a pre-release `obs-frackround` build, remove and re-add the filter because the OBS filter ID changed to `obs_frackground_filter`.

## License

GPL-3.0-or-later. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
