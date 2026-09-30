# Real-Time Collision Avoidance System with Optical Flow Estimation

<img width="610" height="328" alt="pipeline output" src="https://github.com/user-attachments/assets/9b6cb728-117c-4f4e-b7de-2e142a578816" />

## Overview

A camera-based collision warning system for vehicles, built for the AMD Kria
KV260. From a single dash-cam video stream it:

- **detects and tracks** road users with **YOLOv9t**,
- computes **dense optical flow** with **PWC-Net**,
- removes the vehicle's own motion (**ego-motion compensation**, GENEVO),
- estimates each object's **time to collision (TTC)**, either with flow
  heuristics or with an **Echo State Network** that also forecasts TTC
  300 ms ahead,
- raises a warning when the TTC of an object in front of the vehicle drops
  below a threshold.

## Pipeline

```text
Video stream (30 FPS)
    ↓
Object detection (YOLOv9t)
    ↓
Object tracking (Kalman filter + Hungarian matching)
    ↓
Dense optical flow (PWC-Net)
    ↓
Ego-motion correction (GENEVO)
    ↓
TTC estimation (divergence / flow / looming heuristics, or ESN)
    ↓
Collision check against the region in front of the vehicle
    ↓
Annotated output frame
```

## Getting started

Requires Python 3.10-3.12 and, for real-time speed, an NVIDIA GPU.

```bash
git clone https://github.com/shadowPunch/Optic-Flow-Estimation.git
cd Optic-Flow-Estimation
uv venv --python 3.12 && uv pip install -e '.[dev]'    # or: pip install -e '.[dev]'
```

Run it on a video file, or pass a webcam index instead of a file:

```bash
python -m collision_avoidance --source dashcam.mp4
python -m collision_avoidance --source 0
```

Model weights download automatically on first use. Useful options:

| Option | Effect |
|---|---|
| `--ttc-model esn_ttc.pt` | use a trained ESN instead of the heuristic TTC ([how to train it](docs/methods.md#echo-state-network)) |
| `--output out.mp4` | save the annotated video |
| `--no-display --max-frames 300` | headless run with a latency summary |
| `--no-ego` | turn off ego-motion compensation |

While the video plays: `ESC` quit, `P` pause, `E` toggle ego-motion, `S` save
frames, `H` help.

Runs are logged to Weights & Biases. Set `COLLISION_WANDB=0` to run without
an account. Run the tests with `pytest`.

## Results

| | |
|---|---|
| Speed on a laptop GPU (RTX 3050 Ti) | 62 ms per frame (2025 progress report: ~120 ms) |
| TTC error, ESN, now / 300 ms ahead | 14-19 % / 18-22 % median (heuristic: 95 % / 81 %) |
| Detector on the KV260 DPU | whole network on the DPU, 0.317 COCO mAP (original model 0.378) |

The ESN is evaluated on car-rear approach scenarios from the
[EvTTC](https://nail-hnu.github.io/EvTTC/) dataset. Details, baselines and the
per-stage timing breakdown are in [docs/methods.md](docs/methods.md).

## Deploying to the Kria KV260

The models are quantized to INT8 and compiled with **Vitis AI 3.5** for the
KV260's DPU. [deploy/vitis_ai/README.md](deploy/vitis_ai/README.md) has the
Docker-based workflow and the findings:

- **YOLOv9t** needs its SiLU activations swapped for Hardswish before it can be
  compiled at all; after that the entire network runs on the DPU.
- **PWC-Net** compiles (warping and correlation run on the ARM CPU), but its
  INT8 version is not yet accurate enough for TTC.
- **Pruning**: 30 % channel pruning halves the detector's compute but costs
  too much accuracy, so the unpruned Hardswish model is recommended.

Custom FPGA kernels written with Vitis HLS are in [hls/](hls/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `collision_avoidance/` | the pipeline: detection, tracking, flow, ego-motion, TTC, CLI |
| `ttc_esn/` | EvTTC data preparation, ESN training and evaluation |
| `deploy/vitis_ai/` | quantization, compilation and DPU model graphs |
| `hls/` | Vitis HLS kernels |
| `scripts/`, `notebooks/` | benchmarks and YOLO pruning (local or Colab) |
| `docs/` | methods and results, the 2025 progress log |
| `legacy/` | the original 2025 scripts |

## Status and limitations

- The pipeline, TTC models and DPU compilation work and are tested on a PC.
  They have **not yet been run on a KV260 board**, so on-device latency is
  unmeasured.
- The INT8 PWC-Net needs quantization-aware training before it can be used.
- The ESN has only been trained on car-rear scenarios; pedestrians and moving
  targets are untested.
- The HLS kernels are drafts that have not been synthesised.
- This repository was reconstructed in 2026 after the original codebase was
  lost; the original scripts are kept in `legacy/`.

## Contributors

- Shyam B Ganesh (https://github.com/sh-yamm)
- Tanmay S Kushwaha (https://github.com/Tanmay-S-Kushwaha)
- Prateek Ratan (https://github.com/Pratan1)
- Daksh Pandey (https://github.com/D1729)

## Acknowledgments

- **PWC-Net implementation and weights:** [ptlflow](https://github.com/hmorimitsu/ptlflow)
- **TTC dataset:** [EvTTC](https://nail-hnu.github.io/EvTTC/)
- **Ego-motion estimation:** [GENEVO](https://doi.org/10.3390/a18010019)
