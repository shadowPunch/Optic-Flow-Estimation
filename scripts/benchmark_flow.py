"""Host latency of the float optical-flow models (no DPU available).

Compares ptlflow's PWC-DC-Net (reference) with the native DPU-graph
implementation used by the pipeline, with and without the dilated context
network, on GPU and CPU at the pipeline resolution. Results go to W&B
(COLLISION_WANDB=0 to disable).

    python scripts/benchmark_flow.py --runs 50
"""

import argparse
import time

import numpy as np
import ptlflow
import torch
from ptlflow.utils.io_adapter import IOAdapter

from collision_avoidance.pwcnet import PWCNetDPU, load_ptlflow_weights, preprocess
from collision_avoidance.telemetry import start_run

H, W = 384, 512


def time_fn(fn, device: str, runs: int, warmup: int = 5) -> np.ndarray:
    times = []
    for i in range(warmup + runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def variants(device: str):
    rng = np.random.default_rng(0)
    prev, curr = (rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(2))

    ref = ptlflow.get_model("pwcnet", ckpt_path="things").to(device).eval()
    io = IOAdapter(ref.output_stride, (H, W))
    ref_inputs = {k: v.to(device) for k, v in io.prepare_inputs([prev, curr]).items()}
    yield "ptlflow_pwcdcnet", lambda: ref(ref_inputs)

    for dc in (True, False):
        model = load_ptlflow_weights(PWCNetDPU(dc=dc), "things").to(device)
        x = preprocess(prev, curr, to_rgb=dc).to(device)
        yield f"dpu_graph_{'pwcdcnet' if dc else 'pwcnet_nodc'}", lambda m=model, x=x: m(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=50)
    args = p.parse_args()
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    run = start_run("benchmark", {"runs": args.runs, "height": H, "width": W, "devices": devices,
                                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                  "torch": torch.__version__}, tags=["benchmark", "flow", "host", "float"])
    rows = []
    with torch.no_grad():
        for device in devices:
            runs = args.runs if device == "cuda" else max(5, args.runs // 5)
            for name, fn in variants(device):
                t = time_fn(fn, device, runs)
                rows.append([name, device, float(t.mean()), float(np.percentile(t, 50)), float(np.percentile(t, 95))])
                print(f"{name:26s} {device:5s} mean {t.mean():7.1f} ms  p50 {np.percentile(t, 50):7.1f}  p95 {np.percentile(t, 95):7.1f}")
    if run is not None:
        import wandb
        run.log({"flow_latency": wandb.Table(columns=["model", "device", "mean_ms", "p50_ms", "p95_ms"], data=rows)})
        run.summary.update({f"{r[0]}/{r[1]}_mean_ms": r[2] for r in rows})
        run.finish()


if __name__ == "__main__":
    main()
