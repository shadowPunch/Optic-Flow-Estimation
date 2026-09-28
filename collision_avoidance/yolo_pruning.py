"""Structured channel pruning for YOLOv9t (Torch-Pruning + Ultralytics).

YOLOv9t's ELAN blocks split one convolution's output with `chunk(2)`. After
uneven channel pruning the halves no longer line up, so `split_elan_blocks`
first replaces each chunked conv by two convs with the same weights (an exact
rewrite, checked by tests), after which DepGraph can prune freely.

The Detect head is left unpruned so the output layout (and the DPU split in
deploy/vitis_ai/yolo_dpu.py) stays the same. This module is imported when a
pruned checkpoint is unpickled, including inside the Vitis AI docker.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.modules.block import ELAN1, RepNCSPELAN4


def _conv_slice(conv: Conv, start: int, stop: int) -> Conv:
    """A Conv producing output channels [start, stop) of `conv`."""
    part = copy.deepcopy(conv)
    part.conv.weight = nn.Parameter(conv.conv.weight[start:stop].clone())
    part.conv.out_channels = stop - start
    if conv.conv.bias is not None:
        part.conv.bias = nn.Parameter(conv.conv.bias[start:stop].clone())
    if hasattr(conv, "bn"):
        bn = part.bn
        bn.weight = nn.Parameter(conv.bn.weight[start:stop].clone())
        bn.bias = nn.Parameter(conv.bn.bias[start:stop].clone())
        bn.running_mean = conv.bn.running_mean[start:stop].clone()
        bn.running_var = conv.bn.running_var[start:stop].clone()
        bn.num_features = stop - start
    return part


class SplitELAN(nn.Module):
    """ELAN block with the chunked first conv split into two convs (same function)."""

    def __init__(self, block: nn.Module):
        super().__init__()
        half = block.cv1.conv.out_channels // 2
        self.cv1a = _conv_slice(block.cv1, 0, half)
        self.cv1b = _conv_slice(block.cv1, half, 2 * half)
        self.cv2, self.cv3, self.cv4 = block.cv2, block.cv3, block.cv4
        for attr in ("i", "f", "type", "np"):  # Ultralytics graph bookkeeping
            if hasattr(block, attr):
                setattr(self, attr, getattr(block, attr))

    def forward(self, x):
        y = [self.cv1a(x), self.cv1b(x)]
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, 1))


def split_elan_blocks(model: nn.Module) -> nn.Module:
    """Replace every ELAN1 / RepNCSPELAN4 (anywhere in the tree) by SplitELAN, in place."""
    for name, child in model.named_children():
        if isinstance(child, (ELAN1, RepNCSPELAN4)):
            setattr(model, name, SplitELAN(child))
        else:
            split_elan_blocks(child)
    return model


def replace_silu(model: nn.Module, factory=nn.Hardswish) -> nn.Module:
    """Swap SiLU for a DPU-supported activation, in place (needs fine-tuning afterwards).

    DPUCZDX8G cannot run SiLU, so every one of YOLOv9t's 179 activations would
    fall back to the CPU; Hardswish is the closest shape the DPU implements.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            setattr(model, name, factory())
        else:
            replace_silu(child, factory)
    return model


def prune(model: nn.Module, ratio: float, imgsz: int = 640, steps: int = 1) -> nn.Module:
    """Magnitude-based (L2 group norm) channel pruning of all layers except the Detect head."""
    import torch_pruning as tp

    was_training = model.training
    # Trace in eval mode: a train-mode forward would overwrite BatchNorm statistics.
    model = split_elan_blocks(model).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    example = torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
    head = [m for m in model.modules() if isinstance(m, Detect)]
    pruner = tp.pruner.MetaPruner(
        model, example, importance=tp.importance.GroupMagnitudeImportance(p=2),
        iterative_steps=steps, pruning_ratio=ratio, ignored_layers=head, round_to=8,  # multiples of 8 suit the DPU
    )
    for _ in range(steps):
        pruner.step()
    return model.train(was_training)


def complexity(model: nn.Module, imgsz: int = 640) -> dict[str, float]:
    import torch_pruning as tp

    example = torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
    macs, params = tp.utils.count_ops_and_params(model, example)
    return {"gflops": 2 * macs / 1e9, "params_m": params / 1e6}
