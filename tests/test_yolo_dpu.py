"""The DPU split of YOLOv9t must reproduce the Ultralytics model output."""

import sys
from pathlib import Path

import cv2
import pytest
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "vitis_ai"))
import yolo_dpu  # noqa: E402

WEIGHTS = ROOT / "yolov9t.pt"
VIDEO = ROOT / "PWC" / "ptlflow" / "00049.mp4"


@pytest.fixture(scope="module")
def frame():
    if VIDEO.exists():
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
        ok, img = cap.read()
        cap.release()
        if ok:
            return cv2.resize(img, (512, 384))
    pytest.skip("demo video not available")


@pytest.mark.skipif(not WEIGHTS.exists(), reason="yolov9t.pt not present")
def test_split_graph_matches_ultralytics(frame):
    graph = yolo_dpu.load(str(WEIGHTS))
    x = yolo_dpu.preprocess(frame)
    with torch.no_grad():
        reference = YOLO(str(WEIGHTS)).model.float().fuse().eval()(x)[0]
        actual = yolo_dpu.decode(graph(x), graph.head)
    assert actual.shape == reference.shape
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-3)
    detections = yolo_dpu.nms(actual)
    assert len(detections) > 0  # the demo frame contains cars
