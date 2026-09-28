"""Structural correctness of YOLOv9t pruning (accuracy recovery needs fine-tuning, see scripts/prune_yolo.py)."""

import copy
from pathlib import Path

import pytest
import torch

tp = pytest.importorskip("torch_pruning")
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import Conv, Detect  # noqa: E402

from collision_avoidance import yolo_pruning  # noqa: E402

WEIGHTS = Path(__file__).resolve().parents[1] / "yolov9t.pt"
IMGSZ = 320
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="yolov9t.pt not present")


@pytest.fixture(scope="module")
def base():
    return YOLO(str(WEIGHTS)).model.float().eval()


@pytest.fixture(scope="module")
def image():
    return torch.rand(1, 3, IMGSZ, IMGSZ, generator=torch.Generator().manual_seed(0))


def test_split_rewrite_is_exact(base, image):
    split = yolo_pruning.split_elan_blocks(copy.deepcopy(base)).eval()
    with torch.no_grad():
        torch.testing.assert_close(split(image)[0], base(image)[0], rtol=0, atol=1e-5)


def test_removing_dead_channels_preserves_the_function(base, image):
    """Zero each group's channels (BN scale and bias -> SiLU(0) = 0), prune them, compare."""
    model = yolo_pruning.split_elan_blocks(copy.deepcopy(base)).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    bn_of = {m.conv: m.bn for m in model.modules() if isinstance(m, Conv) and hasattr(m, "bn")}
    head = [m for m in model.modules() if isinstance(m, Detect)]
    pruner = tp.pruner.MetaPruner(model, torch.zeros(1, 3, IMGSZ, IMGSZ), pruning_ratio=0.2, ignored_layers=head,
                                  importance=tp.importance.GroupMagnitudeImportance(p=2))
    worst = 0.0
    for group in pruner.step(interactive=True):
        with torch.no_grad():
            for dep, idxs in group:
                if dep.handler == tp.prune_conv_out_channels and dep.target.module in bn_of:
                    bn = bn_of[dep.target.module]
                    bn.weight[idxs] = 0
                    bn.bias[idxs] = 0
            before = model(image)[0].clone()
        group.prune()
        with torch.no_grad():
            worst = max(worst, float((model(image)[0] - before).abs().max()))
    assert worst < 5e-3


def test_prune_reduces_cost_and_keeps_output_layout(base, image):
    pruned = yolo_pruning.prune(copy.deepcopy(base), ratio=0.3, imgsz=IMGSZ).eval()
    before, after = yolo_pruning.complexity(base, IMGSZ), yolo_pruning.complexity(pruned, IMGSZ)
    assert after["gflops"] < 0.7 * before["gflops"]
    assert all(m.conv.out_channels % 8 == 0 for m in pruned.modules() if isinstance(m, Conv) and m not in list(pruned.model[-1].modules()))
    with torch.no_grad():
        assert pruned(image)[0].shape == base(image)[0].shape


def test_replace_silu_swaps_every_activation(base):
    swapped = yolo_pruning.replace_silu(copy.deepcopy(base))
    assert not any(isinstance(m, torch.nn.SiLU) for m in swapped.modules())
    # Ultralytics shares one SiLU instance between layers, so count references, not unique modules.
    silu_refs = sum(isinstance(m, torch.nn.SiLU) for _, m in base.named_modules(remove_duplicate=False))
    assert sum(isinstance(m, torch.nn.Hardswish) for m in swapped.modules()) == silu_refs
