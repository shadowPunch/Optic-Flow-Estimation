"""The DPU-friendly PWC-Net must reproduce ptlflow's output (downloads checkpoints on first run)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ptlflow = pytest.importorskip("ptlflow")
from ptlflow.utils.io_adapter import IOAdapter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "vitis_ai"))
from pwcnet_dpu import PWCNetDPU, load_ptlflow_weights, postprocess, preprocess  # noqa: E402

H, W = 384, 512


def frame_pair():
    """Textured pair with a known shift, so the flow is non-trivial."""
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.uniform(0, 255, (H // 8, W // 8, 3)).astype(np.float32), (W + 16, H + 16), interpolation=cv2.INTER_CUBIC)
    base = base.clip(0, 255).astype(np.uint8)
    return base[8:8 + H, 8:8 + W], base[5:5 + H, 10:10 + W]  # shift (dx=-2, dy=3)


@pytest.mark.parametrize("name,dc", [("pwcnet", True), ("pwcnet_nodc", False)])
def test_matches_ptlflow(name, dc):
    prev, curr = frame_pair()
    reference = ptlflow.get_model(name, ckpt_path="things").eval()
    io = IOAdapter(reference.output_stride, (H, W))
    with torch.no_grad():
        expected = reference(io.prepare_inputs([prev, curr]))["flows"][0, 0]
        model = load_ptlflow_weights(PWCNetDPU(dc=dc), "things")
        actual = postprocess(model(preprocess(prev, curr, to_rgb=dc)))[0]
    epe = torch.linalg.vector_norm(actual - expected, dim=0)
    assert epe.mean() < 1e-3, f"mean EPE vs ptlflow = {epe.mean():.2e}"
    # Sanity: the flow is the known shift in the interior.
    interior = actual[:, 64:-64, 64:-64].mean(dim=(1, 2))
    assert interior.tolist() == pytest.approx([-2.0, 3.0], abs=0.5)
