# Vitis AI 3.5 PyTorch (CPU) + the extra packages our scripts need.
#   docker build -t kria-vitis-ai:3.5 deploy/vitis_ai
FROM xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306

# Keep the numpy/torch builds pytorch_nndct was compiled against: Ultralytics is
# installed without dependency resolution (it would upgrade numpy), and the few
# packages it actually needs are added under constraints.
RUN bash -lc "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch \
    && printf 'numpy==1.22.0\ntorch==1.13.1\ntorchvision==0.14.1\n' > /tmp/constraints.txt \
    && pip install --no-cache-dir --no-deps 'ultralytics==8.3.253' 'ultralytics-thop>=2.0.18' \
    && pip install --no-cache-dir -c /tmp/constraints.txt py-cpuinfo 'polars<1.9' 'wandb<0.20' \
    && python -c 'import ultralytics, wandb, numpy, torch; print(ultralytics.__version__, wandb.__version__, numpy.__version__, torch.__version__)'"
