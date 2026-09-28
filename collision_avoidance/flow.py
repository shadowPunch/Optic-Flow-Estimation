"""Dense optical flow with PWC-DC-Net.

Two interchangeable backends with the same weights (ptlflow "things" checkpoint):
  * native  - `collision_avoidance.pwcnet`, the same graph that is quantized for
              the DPU; ~30 % faster on GPU (default),
  * ptlflow - the reference implementation, kept for comparison.
tests/test_pwcnet.py checks that both produce the same flow.
"""

import cv2
import numpy as np
import torch
from ptlflow.utils import flow_utils

from . import pwcnet


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class FlowEstimator:
    def __init__(self, model_name: str, checkpoint: str, frame_hw: tuple[int, int], device: str, backend: str = "native"):
        self.device = resolve_device(device)
        self.backend = backend
        if backend == "native":
            if frame_hw[0] % 64 or frame_hw[1] % 64:
                raise ValueError(f"native PWC-Net needs frame sides divisible by 64, got {frame_hw}")
            self.dc = model_name == "pwcnet"  # ptlflow naming: 'pwcnet' is PWC-DC-Net
            self.model = pwcnet.load_ptlflow_weights(pwcnet.PWCNetDPU(dc=self.dc), checkpoint).to(self.device)
            torch.backends.cudnn.benchmark = True  # fixed input size
        elif backend == "ptlflow":
            import ptlflow
            from ptlflow.utils.io_adapter import IOAdapter

            self.model = ptlflow.get_model(model_name, ckpt_path=checkpoint).to(self.device).eval()
            self.io = IOAdapter(self.model.output_stride, frame_hw)
        else:
            raise ValueError(f"unknown flow backend '{backend}'")

    @torch.no_grad()
    def __call__(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        """Flow prev -> curr as (H, W, 2) float32 in pixels/frame. Inputs must be frame-sized BGR."""
        if self.backend == "native":
            x = pwcnet.preprocess(prev_bgr, curr_bgr, to_rgb=self.dc).to(self.device)
            flow = pwcnet.postprocess(self.model(x))[0]
        else:
            inputs = {k: v.to(self.device) for k, v in self.io.prepare_inputs([prev_bgr, curr_bgr]).items()}
            flow = self.model(inputs)["flows"][0, 0]
        return flow.permute(1, 2, 0).cpu().numpy()


def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(flow_utils.flow_to_rgb(flow), cv2.COLOR_RGB2BGR)
