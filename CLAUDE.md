# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Rasterman is a desktop GUI app for pixelating and thresholding images, with optional AI background removal (rembg). It uses **pywebview** to host a frameless window whose entire UI is an HTML/CSS/JS frontend backed by a Python `Api` class.

## Running the app

Dependencies are managed with [pixi](https://pixi.sh) (conda-forge + PyPI). Install pixi, then:

```bash
pixi run python main.py
```

There are no tests and no lint commands defined in this project.

## Architecture

The project is deliberately minimal — three source files:

| File | Role |
|------|------|
| `main.py` | Python backend: `Api` class + pywebview window setup |
| `index.html` | Entire frontend — HTML structure, inline styles, inline JS |
| `xp.css` | Windows XP visual theme (third-party stylesheet) |

### Python ↔ JS bridge

pywebview exposes `Api` instance methods to JavaScript as `window.pywebview.api.*`. All calls from JS return Promises. The `Api` object holds mutable state (`current_image_data`, `last_result`, `bg_cache`) for the lifetime of the window.

### Image processing pipeline (`Api.process_image`)

1. **Background removal** — rembg session results are cached in `bg_cache` keyed by model name, so changing threshold/pixelation doesn't re-run the expensive ML step.
2. **Pixelation** — resize down to `result_size` (longest-edge constrained) using the chosen interpolation, then resize back to original dimensions with `INTER_NEAREST`.
3. **Threshold** — convert to grayscale, apply `cv2.threshold`. If the image has an alpha channel (post-background-removal BGRA), alpha is binarized separately and recombined.
4. **Return** — encoded as a base64 PNG data URL for the webview to display.

### Save (`Api.save_image`)

Composites the BGRA result onto a white background before saving as JPEG via a native save dialog.

## Key constraints

- `QT_SCALE_FACTOR_ROUNDING_POLICY` and `QT_ENABLE_HIGHDPI_SCALING` env vars are set at the top of `main.py` **before** any Qt imports — order matters.
- The window is frameless (`frameless=True`); minimize/maximize/close are wired through `window.pywebview.api` calls from the HTML title bar buttons.
- Platform is Linux-only (`platforms = ["linux-64"]` in `pixi.toml`).
