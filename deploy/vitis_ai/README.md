# Vitis AI deployment (KV260 DPU)

Status: **scripts written and host-verified, not yet run in Vitis AI.** The
quantizer and compiler only ship in the Vitis AI docker image, which was not
available when this was reconstructed, and no KV260 board was attached. Every
number below that says "host" was measured on the development laptop; nothing
here has produced an `.xmodel` yet.

## What is here

| File | Purpose | Verified on host |
|---|---|---|
| `../../collision_avoidance/pwcnet.py` | PWC-DC-Net as a plain `nn.Module` (tensor in/out, `nn.Unfold` correlation); shared with the host pipeline, which uses it as its default flow backend | output matches ptlflow `pwcnet` / `pwcnet_nodc` to mean EPE < 1e-3 (`tests/test_pwcnet.py`) |
| `yolo_dpu.py` | YOLOv9t graph up to the Detect-head convolutions; CPU decode + NMS | decoded output matches Ultralytics to 1e-3 (`tests/test_yolo_dpu.py`) |
| `quantize.py` | `inspect` / `calib` / `test` (evaluate vs float + export xmodel) | data sampling and model wrappers smoke-tested; quantizer calls need the docker |
| `compile.sh` | `vai_c_xir` for the KV260 arch + subgraph report | no |

## Fixes relative to the pre-reconstruction attempt (`onnx_pwc/`)

1. **Wrong network.** The pipeline runs ptlflow `pwcnet`, which is
   **PWC-DC-Net** (dilated context network) with the *things* checkpoint. The
   old DPU copy was plain PWC-Net with *sintel* weights.
2. **Double LeakyReLU** after correlation (inside the layer and again in
   `forward`), changing the cost volume.
3. **Missing warp mask**: ptlflow zeroes warped features sampled outside the image.
4. **Channel order**: calibration fed RGB to a model that ptlflow feeds BGR
   (plain PWC-Net) / RGB (PWC-DC-Net). `preprocess(..., to_rgb=model.dc)` now
   encodes this.
5. **Resolution mismatch**: calibration at 256x256, export at 384x448. Both
   now use the pipeline size 384x512 (multiple of 64 as PWC-Net requires).

## Run it

```bash
docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306
docker run -it --rm -v "$PWD":/workspace xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306
# inside the container
conda activate vitis-ai-pytorch
pip install "ultralytics>=8.3,<8.4" wandb     # yolov9t wrapper + tracking (or export COLLISION_WANDB=0)
cd /workspace/deploy/vitis_ai
CLIPS="../../data/evttc/*/video.mp4"           # or any dash-cam clips

for m in pwcnet yolov9t; do
  python quantize.py --model $m --mode inspect --videos $CLIPS   # which ops land on the DPU
  python quantize.py --model $m --mode calib   --videos $CLIPS
  python quantize.py --model $m --mode test    --videos $CLIPS   # prints quantized-vs-float error, writes *_int.xmodel
done
./compile.sh                                   # -> compiled/<model>/<model>_kv260.xmodel + subgraphs.txt
```

EvTTC videos are 2x2 mosaics; for calibration that is acceptable (realistic
road texture), but cropping to the left RGB panel is closer to deployment.

## Expected problems (check the `inspect` report first)

* **PWC-Net partitioning.** `grid_sample` (warping, 4 levels) and the
  unfold-based correlation (5 levels) are not DPUCZDX8G operators, so the
  graph is expected to split into many DPU/CPU segments (one pair per pyramid
  level or more; `inspect` / `subgraphs.txt` will show the real count). The float correlation is
  also the slowest CPU op on the host (the unfold variant takes 251 ms vs
  182 ms for ptlflow's on the laptop CPU), so on the Cortex-A53 it will likely
  dominate. Options: run only the feature pyramid on the DPU, or a DPU-friendly
  correlation (e.g. correlation expressed as grouped 1x1 convolutions over
  shifted inputs).
* **YOLOv9t activations.** The model uses SiLU (237 instances in the old
  quantized graph). DPUCZDX8G does not implement SiLU/sigmoid, so each would
  fall back to the CPU. The usual fix is swapping SiLU for Hardswish or
  LeakyReLU and fine-tuning before quantization (quantization-aware training
  is also available in `pytorch_nndct`).
* **DPU target name.** `quantize.py` uses `DPUCZDX8G_ISA1_B4096`; confirm it
  matches the bitstream loaded on the board with `xdputil query`.

## Host float benchmark (`scripts/benchmark_flow.py`, RTX 3050 Ti laptop, 384x512)

| Model | GPU mean | CPU mean |
|---|---|---|
| ptlflow PWC-DC-Net (reference) | 42.9 ms | 182 ms |
| DPU-graph PWC-DC-Net (pipeline default) | 29.8 ms | 251 ms |
| DPU-graph PWC-Net (no DC) | 25.1 ms | 209 ms |

W&B run: `kria-collision-avoidance/uxtpuu6y`.

## Not reconstructed

* **Pruning** (Vitis AI Optimizer) - no trace of the original configuration.
* **On-board runner** (VART / Vitis AI Library app) and the original
  "72 % inference-time reduction" measurement - need the board.
