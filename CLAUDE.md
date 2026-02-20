# CLAUDE.md — Lada Project Guide

## Project Overview

Lada is a tool for recovering pixelated (mosaic) regions in adult videos (JAV). It uses YOLO for mosaic detection and BasicVSR++ for temporal video restoration. Supports CLI, GUI (GTK4), multi-GPU parallel processing, and multiple platforms (Linux, Windows, Docker, Flatpak).

**License:** AGPL-3.0

## Quick Reference

### Running the CLI

```shell
# Activate venv first
source .venv/bin/activate

# Single GPU (default)
lada-cli --input video.mp4

# Multi-GPU
lada-cli --input video.mp4 --num-gpus 4

# Highest quality
lada-cli --input video.mp4 --num-gpus 4 --mosaic-detection-model v4-accurate --encoding-preset hevc-nvidia-gpu-uhq --max-clip-length 360

# Or run directly without activating venv
.venv/bin/python -m lada.cli.main --input video.mp4
```

### Running the GUI

```shell
lada
# or
.venv/bin/python -m lada.gui.main
```

## Project Structure

```
lada/
├── cli/                    # CLI entry point (main.py) and utilities
├── gui/                    # GTK4 GUI application
├── models/                 # ML model definitions
│   ├── basicvsrpp/         # BasicVSR++ restoration model
│   ├── deepmosaics/        # DeepMosaics restoration model (legacy)
│   └── yolo/               # YOLO11 segmentation for mosaic detection
├── multigpu/               # Multi-GPU parallel processing
│   ├── coordinator.py      # Orchestrates segment splitting, worker spawning, merging
│   └── merger.py           # FFmpeg-based segment trimming and concatenation
├── restorationpipeline/    # Core restoration pipeline
│   ├── __init__.py         # load_models() — loads detection + restoration models
│   ├── frame_restorer.py   # FrameRestorer — main pipeline (threaded producer/consumer)
│   └── mosaic_detector.py  # MosaicDetector — YOLO inference + scene/clip creation
├── utils/                  # Shared utilities
│   ├── __init__.py         # VideoMetadata dataclass, type aliases (ImageTensor, etc.)
│   ├── video_utils.py      # VideoReader, VideoWriter, ffprobe metadata, encoding presets
│   ├── audio_utils.py      # Audio muxing via ffmpeg
│   ├── threading_utils.py  # PipelineQueue, PipelineThread, EOF/STOP markers
│   ├── image_utils.py      # Resize, pad/unpad operations
│   └── os_utils.py         # Platform detection, subprocess startup info
├── datasetcreation/        # Dataset creation tools (training only)
└── locale/                 # i18n translation files (.mo)
```

## Architecture

### Restoration Pipeline (single GPU)

The pipeline is threaded with producer/consumer queues:

1. **Frame Feeder** (MosaicDetector) — reads video frames, batches them
2. **YOLO Inference** (MosaicDetector) — runs detection model, produces frame detections
3. **Frame Detector** (MosaicDetector) — groups detections into Scenes, creates Clips
4. **Clip Restoration** (FrameRestorer) — runs BasicVSR++ on mosaic clips
5. **Frame Restoration** (FrameRestorer) — blends restored clips back into original frames
6. **Output** — VideoWriter encodes frames via PyAV

Key classes:
- `Scene`: Groups of consecutive frames with overlapping mosaic bounding boxes
- `Clip`: A Scene cropped/resized to 256x256 for the restoration model
- `FrameRestorer`: Iterable that yields `(restored_frame, pts)` tuples

### Multi-GPU Pipeline

When `--num-gpus > 1`:
1. Video is split into N segments with overlap (default 60 frames)
2. Each segment is processed by a separate OS process on a different GPU (`torch.multiprocessing` spawn context)
3. Each worker loads its own models and runs the full single-GPU pipeline
4. Segments are trimmed at overlap midpoints and concatenated via ffmpeg
5. Audio is remuxed from the original file

### Threading Model

- `PipelineQueue` wraps `queue.Queue` with stats tracking and named debugging
- `PipelineThread` wraps `threading.Thread` with error propagation
- `EOF_MARKER` / `STOP_MARKER` sentinel objects control pipeline shutdown
- Error in any thread triggers graceful shutdown of all threads

## Key Conventions

### Code Style
- Python 3.12+ (uses `X | Y` union types, not `Optional[X]`)
- No type stubs or extensive docstrings — code is self-documenting
- Uses `gettext` for i18n: user-facing strings wrapped in `_()`
- Logging via `logging` module, level controlled by `LOG_LEVEL` env var

### Video I/O
- **Reading**: PyAV (`av` package) via `VideoReader` class — handles corrupt frames
- **Writing**: PyAV via `VideoWriter` class — handles PTS reordering
- **Metadata**: `ffprobe` via subprocess for video metadata
- **Audio muxing**: `ffmpeg` via subprocess (stream copy when possible)
- Frames are `torch.Tensor` in BGR format (OpenCV convention)

### Model Weights
- Stored in `model_weights/` directory (or `LADA_MODEL_WEIGHTS_DIR` env var)
- Detection models: `.pt` files (YOLO format)
- Restoration models: `.pth` files (PyTorch state dict)
- Default detection: `v4-fast`, default restoration: `basicvsrpp-v1.2`

### Dependencies
- Package manager: `uv` (see `pyproject.toml` for index configuration)
- PyTorch installed from specific indexes per GPU vendor (nvidia/intel/cpu)
- Pinned versions: `ultralytics==8.4.4`, `mmengine==0.10.7` (patched)
- `opencv-python==4.12.0.88` (4.13 has GUI text bugs)

## Build & Development

```shell
# Install with uv (Nvidia GPU)
uv sync --extra nvidia --extra gui

# Install with pip
pip install -e ".[nvidia,gui]"

# Run tests (if any)
python -m pytest

# Entry points defined in pyproject.toml
# lada -> lada.gui.main:main
# lada-cli -> lada.cli.main:main
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LADA_MODEL_WEIGHTS_DIR` | Custom path for model weights (default: `model_weights/`) |
| `LOG_LEVEL` | Python logging level (default: `WARNING`) |
| `TMPDIR` | Temp directory for intermediate video files |

## Common Pitfalls

- **Multiprocessing with CUDA**: Must use `spawn` context (not `fork`) and create shared `Value`/`Lock` objects from the same context
- **Frame PTS ordering**: Some videos have non-monotonic PTS; `VideoWriter` uses a min-heap to reorder
- **Queue deadlocks**: Pipeline shutdown must unblock both producers and consumers in correct order
- **Corrupt video frames**: `VideoReader` duplicates the last good frame on decode errors
- **i18n**: All user-facing CLI strings must use `_()` for translation support
