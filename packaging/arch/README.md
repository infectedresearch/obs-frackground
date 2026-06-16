# Arch/AUR Packaging

This directory contains the source package files for the AUR package `obs-frackground`.

Build locally with:

```sh
makepkg -f
```

Install locally with:

```sh
makepkg -si
```

Regenerate AUR metadata after editing `PKGBUILD`:

```sh
makepkg --printsrcinfo > .SRCINFO
```

The package depends on the virtual `onnxruntime` package. Users can satisfy it with `onnxruntime-cpu`, `onnxruntime-cuda`, or another compatible provider.
