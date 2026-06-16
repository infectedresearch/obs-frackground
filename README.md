# obs-frackround

`obs-frackround` is a Linux-first OBS Studio filter plugin for high-quality person matting and transparent background output.

The project goal is to outperform generic background-removal plugins by using a GPU-first, video-aware matting pipeline with conservative foreground preservation.

Current status: initial OBS filter scaffold with a test matte shader. ONNX Runtime CUDA and RVM-style model inference are the next implementation step.

The current build can create an ONNX Runtime session asynchronously from a selected `.onnx` model path and prefers CUDA when available. Frame preprocessing and matte compositing from model output are not implemented yet.

## Goals

- OBS Studio 32.x support.
- Transparent background output first.
- ONNX Runtime CUDA inference backend.
- Async inference that does not block OBS rendering.
- Soft alpha matting instead of hard binary masks.
- Swappable model backend for future custom or distilled models.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

## Download Test Model

```sh
bash scripts/download-rvm-models.sh
```

This downloads the official RVM MobileNetV3 FP16 ONNX model into `models/`. The model file is ignored by git.

If present, this model is installed beside the plugin and used as the default model path.

## Local Install

```sh
cmake --install build --prefix ~/.local
```

Then restart OBS and add the `Frackround Background Matting` filter to a video source.

Use `Matting model (.onnx)` to select an ONNX matting model. Model loading happens on a worker thread so OBS rendering is not blocked.

## Current Frame Path

The current implementation submits frames to RVM from OBS `filter_video` when the source provides one of these raw formats:

- `RGBA`
- `BGRA`
- `BGRX`
- `NV12`
- `I420`
- `YUY2`
- `UYVY`
- `YVYU`

Inference input is resized while preserving source aspect ratio. The matte is uploaded as an 8-bit GPU texture and applied by the OBS shader.

GPU-only render sources that do not pass through `filter_video` still need a texture readback path.

The plugin also has a render-texture fallback: GPU-rendered sources are rendered into an offscreen OBS texture, double-buffered through staging surfaces, submitted to the RVM worker, and drawn through the matte shader. Sources that already provide raw frames skip this readback path to avoid duplicate inference work.

## Ready-To-Test Defaults

- Model: `models/rvm_mobilenetv3_fp16.onnx`
- Backend: CUDA enabled
- Quality mode: `Balanced`
- Inference size: `512x512`
- Foreground protection: `0.75`
- Edge softness: `0.35`
- Temporal smoothing: `0.70`

## Runtime Plan

The initial production pipeline will use:

- ONNX Runtime CUDA for inference.
- An RVM-style video matting model.
- A worker thread that drops stale frames and reuses the latest valid matte.
- OBS GPU shader compositing for alpha output.

## License

GPL-3.0-or-later. See `LICENSE`.
