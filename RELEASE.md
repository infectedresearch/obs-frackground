# Release Notes

## v0.1.0-alpha.1

Initial Linux alpha for `obs-frackground`.

### Scope

- Linux x86_64 OBS Studio filter plugin.
- Background removal / auto greenscreen with transparent output.
- ONNX Runtime inference using the bundled RobustVideoMatting MobileNetV3 FP16 model.
- CUDA preference with CPU fallback.

### Artifact Install

Extract the release archive into your home directory:

```sh
tar -xzf obs-frackground-v0.1.0-alpha.1-linux-x86_64.tar.gz -C ~/.local --strip-components=1
```

Then restart OBS and add `Frackground Background Removal` as a filter on a video source.

### Expected Archive Layout

```text
obs-frackground/
├── lib/obs-plugins/obs-frackground.so
└── share/obs/obs-plugins/obs-frackground/
    ├── frackground.effect
    ├── locale/en-US.ini
    └── rvm_mobilenetv3_fp16.onnx
```

### Test Checklist

- Clean configure and build with CMake.
- Run `frackground-probe models/rvm_mobilenetv3_fp16.onnx`.
- Install locally and verify OBS lists `Frackground Background Removal`.
- Test with a V4L2 camera source.
- Test with a media source that provides raw frames.
- Toggle debug mask preview.
- Verify CUDA path when available and CPU fallback when CUDA is unavailable.

### Known Limitations

- Alpha release; OBS/libobs and ONNX Runtime ABI compatibility are distro-sensitive.
- The Linux binary artifact is not a universal package.
- CPU inference is expected to be slow for real-time use.
- Only RVM-style ONNX models with expected input/output names are supported.
- GPU-rendered sources that do not provide raw frames are not considered supported in this alpha.
- Users upgrading from pre-release `obs-frackround` builds must re-add the OBS filter because the filter ID changed.
