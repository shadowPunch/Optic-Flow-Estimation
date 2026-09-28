"""Object detection with YOLOv9t, filtered to collision-relevant COCO classes."""

import numpy as np
from ultralytics import YOLO

from .config import RELEVANT_CLASSES
from .tracking import Detection


class Detector:
    def __init__(self, weights: str, conf_threshold: float, device: str):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.device = device

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self.model(frame_bgr, verbose=False, device=self.device)[0]
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            conf = float(box.conf.item())
            cls = int(box.cls.item())
            if conf >= self.conf_threshold and cls in RELEVANT_CLASSES:
                bbox = box.xyxy.cpu().numpy().astype(int).flatten()
                detections.append(Detection(bbox, RELEVANT_CLASSES[cls], conf))
        return detections
