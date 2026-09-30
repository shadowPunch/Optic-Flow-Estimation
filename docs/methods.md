# Methods and detailed results

Technical reference for the pipeline stages, the TTC models and the host
measurements. The deployment work (quantization, pruning, KV260 compilation)
is documented in [deploy/vitis_ai/README.md](../deploy/vitis_ai/README.md).

## Pipeline architecture

```text
camera frame (resized to 512x384)
   |
   +--> PWC-DC-Net optical flow ("things" weights) --------------------+
   |                                                                   |
   +--> ego-motion: genetic search for (dx, dy, theta) every 15 frames |
   |       -> subtract induced flow  <---------------------------------+
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

All tunables live in `collision_avoidance/config.py`; the defaults are the
constants of the original 2025 script (`legacy/ttc-video_feed.py`).

## TTC estimation

### Heuristic (original)

For each tracked box, three estimates are fused with weights 3:2:1:

* **divergence**: `1 / (median div(flow) * fps)`;
* **flow magnitude**: pinhole distance from an assumed 1.5 m object and
  f = 500 px, speed from the mean flow magnitude;
* **looming**: radial flow around the box centre relative to box size.

It is kept exactly as originally tuned (a parity test compares it with the
original script), because the warning thresholds depend on it. Known
limitations: the divergence of an approaching plane is `2/TTC`, so the first
estimate is biased low by 2x; in the second, the assumed distance cancels out
(`TTC = f / (|flow| * fps)`), so it measures lateral motion as much as approach.

### Echo State Network

* **Features** (`ttc_esn/features.py`): 17 per object per frame - divergence
  and flow-magnitude statistics, mean flow, a least-squares expansion rate
  (flow = t + a (p - c)) and its residual, bbox growth rate, bbox geometry, and
  the heuristic inverse TTCs. Rates are per second, so a model trained at
  20 FPS applies at 30 FPS.
* **Model** (`ttc_esn/model.py`): a frozen 300-unit leaky reservoir (spectral
  radius 0.9, leak 0.3, 10 % density) and a trained 2x64 MLP readout on
  `[features, state]`, predicting log TTC at +0, 0.1, 0.2 and 0.3 s. The live
  pipeline keeps one reservoir state per track (`ttc_esn/online.py`).
* **Data** (`ttc_esn/evttc.py`, `extract.py`): five
  [EvTTC](https://nail-hnu.github.io/EvTTC/) car-rear sequences (CCRs-1
  low/medium/high, CCRs-2 low/high; 20 FPS video, 100 Hz ground truth). The
  deployed pipeline is run over the left RGB camera and the target track is
  found from the dataset annotations. Video and ground truth are aligned per
  sequence with the rigid-target constraint `bbox_width * distance = const`
  (offsets 0 to -0.17 s, residual spread 1-3 %).

Results, leave-one-sequence-out cross-validation (705 test frames). Metric:
relative TTC error `|pred - gt| / gt`. Baselines are extrapolated to future
horizons as `TTC(t) - h`.

| Method | median, now | median, +300 ms | within 20 %, +300 ms |
|---|---|---|---|
| Heuristic (original) | 95 % | 81 % | 4 % |
| 1 / expansion rate (single physical feature) | 20 % | 23 % | 47 % |
| MLP on features, no reservoir (3 seeds) | 23-26 % | 26-29 % | 37-43 % |
| **ESN + MLP (3 seeds)** | **14-19 %** | **18-22 %** | 47-53 % |

The ESN is 4-5x more accurate than the heuristic and the reservoir helps on
every seed, but its lead over the single expansion-rate feature is modest.
Pedestrian and moving-target scenarios are not covered. In the live pipeline
the ESN costs 5.6 ms per frame.

Reproduce: `python -m ttc_esn.evttc && python -m ttc_esn.extract && python -m ttc_esn.train`
(add `--variant mlp` for the no-reservoir ablation).

## Ego-motion compensation

A GENEVO-style genetic search ([Algorithms 18(1):19](https://doi.org/10.3390/a18010019))
finds the rigid 2-D motion `(dx, dy, theta)` that best aligns consecutive
grayscale frames, converts it to the flow it induces and subtracts it.
Changes from the original script, all covered by tests:

1. theta was sampled from +-3 rad; it is now +-0.05 rad with its own mutation
   scale, as in the HLS port;
2. the rotational part of the induced flow had the wrong sign;
3. the search runs on half-resolution frames, about 3x cheaper.

It models translation and in-plane rotation only, not the forward-motion
expansion that dominates in driving.

## Host latency

300 frames of a Nexar dash-cam clip, RTX 3050 Ti laptop GPU, 512x384.
"2025 settings" is the refactored pipeline with the original configuration;
"current" uses the native PWC-Net backend and the half-resolution ego-motion
search.

| Stage | 2025 settings, mean | current, mean | current, p95 |
|---|---|---|---|
| PWC-DC-Net flow | 48.5 ms | 35.4 ms | 36.6 ms |
| ego-motion (search every 15th frame) | 22.7 ms | 8.9 ms | 83.5 ms |
| YOLOv9t detection | 12.3 ms | 13.1 ms | 16.3 ms |
| tracking | 0.3 ms | 0.3 ms | 0.4 ms |
| TTC (heuristic) | 2.8 ms | 3.4 ms | 5.1 ms |
| **total** | **87.1 ms** | **61.8 ms** | **132 ms** |

The 2025 progress report measured ~120 ms per frame against a 30-40 ms target.
`flow_backend="ptlflow"` switches back to the reference flow implementation
(identical output, `tests/test_pwcnet.py`).

## Data and weights

| Item | Source |
|---|---|
| `yolov9t.pt` | Ultralytics YOLOv9t (COCO), in the repository |
| PWC-DC-Net "things" | ptlflow release `pwcdcnet-things-cc223701.ckpt`, downloaded on first use |
| EvTTC sequences | `python -m ttc_esn.evttc` (links in `ttc_esn/evttc_manifest.json`) |
| ESN weights | trained with `ttc_esn.train` into `outputs/models/` |
| Demo clips | Nexar dash-cam collision dataset (Kaggle) |

Every training, evaluation and inference run is logged to Weights & Biases
(project `kria-collision-avoidance`).
