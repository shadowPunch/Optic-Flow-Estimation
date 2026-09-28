"""The native (DPU-graph) PWC-Net must reproduce ptlflow's output (downloads checkpoints on first run)."""

import cv2
import numpy as np
import pytest
import torch

ptlflow = pytest.importorskip("ptlflow")
from ptlflow.utils.io_adapter import IOAdapter  # noqa: E402

from collision_avoidance.flow import FlowEstimator  # noqa: E402
from collision_avoidance.pwcnet import PWCNetDPU, load_ptlflow_weights, postprocess, preprocess  # noqa: E402

H, W = 384, 512
SHIFT = (-2.0, 3.0)  # (dx, dy) of the synthetic pair


def frame_pair():
    """Textured pair with a known shift, so the flow is non-trivial."""
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.uniform(0, 255, (H // 8, W // 8, 3)).astype(np.float32), (W + 16, H + 16), interpolation=cv2.INTER_CUBIC)
    base = base.clip(0, 255).astype(np.uint8)
    return base[8:8 + H, 8:8 + W], base[5:5 + H, 10:10 + W]


@pytest.mark.parametrize("name,dc", [("pwcnet", True), ("pwcnet_nodc", False)])
def test_model_matches_ptlflow(name, dc):
    prev, curr = frame_pair()
    reference = ptlflow.get_model(name, ckpt_path="things").eval()
    io = IOAdapter(reference.output_stride, (H, W))
    with torch.no_grad():
        expected = reference(io.prepare_inputs([prev, curr]))["flows"][0, 0]
        model = load_ptlflow_weights(PWCNetDPU(dc=dc), "things")
        actual = postprocess(model(preprocess(prev, curr, to_rgb=dc)))[0]
    epe = torch.linalg.vector_norm(actual - expected, dim=0)
    assert epe.mean() < 1e-3, f"mean EPE vs ptlflow = {epe.mean():.2e}"
    interior = actual[:, 64:-64, 64:-64].mean(dim=(1, 2))
    assert interior.tolist() == pytest.approx(SHIFT, abs=0.5)


def test_pipeline_flow_backends_agree():
    prev, curr = frame_pair()
    native = FlowEstimator("pwcnet", "things", (H, W), "cpu", backend="native")(prev, curr)
    reference = FlowEstimator("pwcnet", "things", (H, W), "cpu", backend="ptlflow")(prev, curr)
    assert native.shape == reference.shape == (H, W, 2)
    assert np.linalg.norm(native - reference, axis=-1).mean() < 1e-3


def test_native_backend_rejects_unsupported_size():
    with pytest.raises(ValueError):
        FlowEstimator("pwcnet", "things", (380, 512), "cpu", backend="native")
