"""Dense optical flow with PWC-Net through ptlflow."""

import cv2
import numpy as np
import ptlflow
import torch
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class FlowEstimator:
    """Wraps a ptlflow model; `ptlflow.get_model('pwcnet')` is PWC-DC-Net."""

    def __init__(self, model_name: str, checkpoint: str, frame_hw: tuple[int, int], device: str):
        self.device = resolve_device(device)
        self.model = ptlflow.get_model(model_name, ckpt_path=checkpoint).to(self.device).eval()
        self.io = IOAdapter(self.model.output_stride, frame_hw)

    @torch.no_grad()
    def __call__(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        """Flow prev -> curr as (H, W, 2) float32 in pixels/frame. Inputs must be frame-sized BGR."""
        inputs = self.io.prepare_inputs([prev_bgr, curr_bgr])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        flows = self.model(inputs)["flows"]
        return flows[0, 0].permute(1, 2, 0).cpu().numpy()


def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(flow_utils.flow_to_rgb(flow), cv2.COLOR_RGB2BGR)
