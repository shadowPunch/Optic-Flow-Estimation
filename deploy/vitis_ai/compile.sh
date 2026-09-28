#!/usr/bin/env bash
# Compile the quantized xmodels for the KV260 DPU. Run inside the Vitis AI 3.5 docker
# after `quantize.py --mode test` has produced quantized/<model>/*_int.xmodel.
set -euo pipefail

ARCH=${ARCH:-/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json}
IN_DIR=${IN_DIR:-quantized}
OUT_DIR=${OUT_DIR:-compiled}

for model in pwcnet yolov9t; do
  xmodel=$(ls "$IN_DIR/$model"/*_int.xmodel)
  vai_c_xir --xmodel "$xmodel" --arch "$ARCH" --output_dir "$OUT_DIR/$model" --net_name "${model}_kv260"
  # How the graph was partitioned: every CPU subgraph is a DPU<->CPU round trip at runtime.
  xir subgraph "$OUT_DIR/$model/${model}_kv260.xmodel" | tee "$OUT_DIR/$model/subgraphs.txt"
done
