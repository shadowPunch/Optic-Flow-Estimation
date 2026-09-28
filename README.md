# Real-Time Collision Avoidance with Optical Flow (AMD Kria KV260)

<img width="610" height="328" alt="pipeline output" src="https://github.com/user-attachments/assets/9b6cb728-117c-4f4e-b7de-2e142a578816" />

A monocular collision-warning pipeline for vehicles: YOLOv9t finds and tracks
road users, PWC-Net estimates dense optical flow, ego-motion is compensated,
and a time-to-collision (TTC) is estimated per object, either with the
original flow heuristics or with an Echo State Network (ESN) that also
forecasts TTC 300 ms ahead. The target platform is the AMD Kria KV260 (DPU via
Vitis AI, custom kernels via Vitis HLS).

> **Reconstruction note (Sept 2026).** The original 2025 codebase was lost.
> This repository was rebuilt from what survived: the pipeline script
> (`legacy/`), HLS drafts, and partial Vitis AI quantization output. The
> table below separates what is original, what was reconstructed and
> verified, and what is still missing.

## Status

| Component | State |
|---|---|
| Python pipeline (detection, tracking, flow, ego-motion, TTC, ROI warnings) | **Original, refactored** into `collision_avoidance/`; numerically identical TTC (parity test against `legacy/`) |
| Heuristic TTC (divergence / flow magnitude / looming, 3:2:1) | Original |
| ESN + MLP TTC model, 0-300 ms horizons | **Rebuilt** (`ttc_esn/`); original code, data prep and weights were lost - see [Results](#esn-ttc-results) |
| PWC-Net / YOLOv9t DPU graphs | **Rebuilt and verified on host** against ptlflow / Ultralytics (`deploy/vitis_ai/`) |
| Vitis AI quantization + KV260 compilation | Scripts written, **not run** (no Vitis AI docker) |
| Pruning, on-board DPU runner, "72 % inference-time reduction" | **Missing** - need Vitis AI Optimizer and the board |
| HLS kernels (`hls/`) | Original drafts; not synthesised, known issues listed in [hls/README.md](hls/README.md) |

## Architecture

```text
camera frame (resized to 512x384)
   |
   +--> PWC-DC-Net optical flow (ptlflow, "things" weights) ----------+
   |                                                                  |
   +--> ego-motion: genetic search for (dx, dy, theta) every 15 frames |
   |       -> subtract induced flow  <--------------------------------+
   |
   +--> YOLOv9t detection (collision-relevant COCO classes)
   |       -> Kalman (constant velocity) + Hungarian IoU tracking,
   |          flow-guided propagation through missed detections
   |
   +--> per track: TTC estimator
   |       heuristic (default)  or  ESN: 17 flow features -> reservoir -> MLP
   |
   +--> ROI check: warn if TTC <= 1.12 s inside ROI, <= 0.56 s outside
```

| Module | Responsibility |
|---|---|
| `collision_avoidance/config.py` | All tunables; defaults are the original script's constants |
| `collision_avoidance/pipeline.py` | Per-frame orchestration and stage timing (no I/O, testable) |
| `collision_avoidance/flow.py`, `detection.py`, `tracking.py`, `ego_motion.py`, `ttc.py` | One stage each |
| `collision_avoidance/app.py` | CLI: video/webcam, display, recording, latency summary |
| `collision_avoidance/telemetry.py` | Weights & Biases runs for every entry point |
| `ttc_esn/` | EvTTC download + sync, feature extraction, ESN model, training/CV, pipeline adapter |
| `deploy/vitis_ai/` | DPU-friendly model graphs, quantize/compile scripts, runbook |
| `hls/` | Vitis HLS kernels and C++ ports (drafts) |
| `legacy/` | Original scripts, verbatim, for provenance and parity tests |
| `docs/progress-report-2025.md` | The team's weekly log from 2025 |

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]'   # CUDA 12.4 wheels; ptlflow needs Python < 3.13
# exact versions used for the reported runs: requirements-lock.txt
.venv/bin/python -m pytest                              # 23 tests

# Run on a video (or --source 0 for a webcam). Weights download on first use.
.venv/bin/python -m collision_avoidance --source clip.mp4
.venv/bin/python -m collision_avoidance --source clip.mp4 --ttc-model outputs/models/esn_ttc.pt
.venv/bin/python -m collision_avoidance --source clip.mp4 --no-display --max-frames 300 --output out.mp4
```

Keys: `ESC` quit, `P` pause, `R` real-time pacing, `E` toggle ego-motion, `S` save frames, `H` help.

Every run is logged to Weights & Biases (project `kria-collision-avoidance`);
set `COLLISION_WANDB=0` to run offline. If tracking is enabled and W&B cannot
be reached, runs stop with an error instead of silently going untracked.

## Host performance

Headless run of `00049.mp4` (Nexar dash-cam clip), 300 frames, RTX 3050 Ti
laptop GPU (W&B run `z14jtv2f`):

| Stage | mean | p50 | p95 |
|---|---|---|---|
| PWC-DC-Net flow | 48.5 ms | 47.5 ms | 54.3 ms |
| ego-motion (search every 15th frame) | 22.7 ms | 3.6 ms | 288 ms |
| YOLOv9t detection | 12.3 ms | 12.2 ms | 13.2 ms |
| tracking | 0.3 ms | 0.3 ms | 0.4 ms |
| TTC (heuristic) | 2.8 ms | 2.7 ms | 4.1 ms |
| **total** | **87.1 ms** | **66.6 ms** | **352 ms** |

The 2025 report measured ~120 ms/frame against a 30-40 ms target. Flow is
the bottleneck; the ego-motion search causes the periodic spikes. Flow-model
comparisons (including the DPU graph variants) are in
[deploy/vitis_ai/README.md](deploy/vitis_ai/README.md).

## TTC estimation

### Heuristic (original)

For each tracked box, three estimates are fused with weights 3:2:1:

* **divergence**: `1 / (median div(flow) * fps)`,
* **flow magnitude**: pinhole distance from an assumed 1.5 m object and
  f = 500 px, speed from the mean flow magnitude,
* **looming**: radial flow around the box centre relative to box size.

Kept exactly as originally tuned, since the warning thresholds depend on it.
Known limitations: the divergence of an approaching plane is `2/TTC`, so the
first estimate is biased low by 2x; in the second, the assumed distance
cancels out (`TTC = f / (|flow| * fps)`), so it measures lateral motion as
much as approach. On EvTTC its median relative error is large (see below),
which motivated the ESN.

### ESN (rebuilt)

* **Features (17, per object per frame, `ttc_esn/features.py`)**: divergence
  statistics, flow magnitude statistics, mean flow, a least-squares expansion
  rate (flow = t + a (p - c)), its residual, bbox growth rate, bbox geometry,
  and the heuristic inverse TTCs. All rates are per second, so a model
  trained at 20 FPS applies at 30 FPS.
* **Model (`ttc_esn/model.py`)**: 300-unit leaky reservoir (spectral radius
  0.9, leak 0.3, 10 % density), frozen; a 2x64 MLP readout trained on
  `[features, state]` to predict log TTC at +0, 0.1, 0.2 and 0.3 s. Only the
  readout is trained. One reservoir state is kept per track in the live
  pipeline (`ttc_esn/online.py`).
* **Data (`ttc_esn/evttc.py`, `extract.py`)**: five EvTTC car-rear sequences,
  CCRs-1 low/medium/high and CCRs-2 low/high (20 FPS video, 100 Hz ground
  truth), listed in `ttc_esn/evttc_manifest.json`. Moving-car and pedestrian
  scenarios are out of scope for this rebuild; add them to the manifest to
  extend it. The deployed pipeline itself is run over the left RGB camera; the
  target track is identified from the dataset's annotations by IoU and
  track-id continuity. Video and ground truth are aligned per sequence with the
  rigid-target constraint `bbox_width * distance = const` (fitted offsets
  0 to -0.17 s, residual spread 1-3 %; implied target widths 2.3 m for CCRs-1
  and 1.8 m for CCRs-2, consistent within each target car).

#### ESN TTC results

Leave-one-sequence-out cross-validation over the 5 sequences (each sequence is
the test set once; 705 test frames at +0 s, 687 at +0.3 s). Metric: relative
TTC error |pred - gt| / gt (the EvTTC metric). Baselines are extrapolated
to future horizons assuming constant closing speed (TTC(t) - h).

| Method | median, now | median, +300 ms | mean, +300 ms | within 20 %, +300 ms |
|---|---|---|---|---|
| Legacy heuristic | 95 % | 81 % | 134 % | 4 % |
| 1 / expansion rate (physics, single feature) | 20 % | 23 % | 27 % | 47 % |
| MLP on features, no reservoir (3 seeds) | 23-26 % | 26-29 % | 38-45 % | 37-43 % |
| **ESN + MLP (3 seeds)** | **14-19 %** | **18-22 %** | 27-28 % | 47-53 % |

What this supports:

* The learned model is ~4-5x more accurate than the original heuristic.
* The reservoir matters: with identical features and readout, removing it
  worsens every seed.
* The gain over the best single physical feature is modest (a few points of
  median error, equal on mean error) and comes from 5 closely related
  scenarios. The "accurate trajectory prediction 300 ms ahead" claim should be
  read as ~20 % median error at +300 ms on car-rear approaches; pedestrians
  and moving targets are untested.

The shipped model is `outputs/models/esn_ttc.pt` (seed 0, fixed before
seeing results; W&B `final-esn` run `ny714n30`, CV run `02k1obif`). In the
live pipeline it costs 5.6 ms/frame on the laptop (W&B `zbup23uf`).

## Ego-motion compensation

A GENEVO-style genetic search ([Algorithms 18(1):19](https://doi.org/10.3390/a18010019))
finds the rigid 2-D motion `(dx, dy, theta)` that best aligns consecutive
grayscale frames, converts it to the flow it induces and subtracts it. Two
deviations from the original script, both covered by tests:

1. theta was sampled from +-3 rad (the translation range); it is now
   +-0.05 rad with its own mutation scale, as in the HLS port;
2. the rotational part of the induced flow had the wrong sign for OpenCV's
   rotation matrix.

It models translation and in-plane rotation only, not the forward-motion
expansion that dominates in driving.

## Data and weights

| Item | Source | In git |
|---|---|---|
| `yolov9t.pt` | Ultralytics YOLOv9t (COCO) | yes |
| PWC-DC-Net "things" | ptlflow release `pwcdcnet-things-cc223701.ckpt`, auto-downloaded | no |
| EvTTC subset (5 sequences) | `python -m ttc_esn.evttc` (links in `ttc_esn/evttc_manifest.json`) | no (`data/`) |
| ESN weights | `python -m ttc_esn.extract && python -m ttc_esn.train` -> `outputs/models/` | no |
| Demo clips `000xx.mp4` | Nexar dash-cam collision dataset (Kaggle), local copies only | no |

## Credits

Other contributors: [Shyam B Ganesh](https://github.com/sh-yamm), [Tanmay S Kushwaha](https://github.com/Tanmay-S-Kushwaha),
[Prateek Ratan](https://github.com/Pratan1), [Daksh Pandey](https://github.com/D1729).
PWC-Net implementation and weights: [ptlflow](https://github.com/hmorimitsu/ptlflow).
TTC data: [EvTTC](https://nail-hnu.github.io/EvTTC/).
