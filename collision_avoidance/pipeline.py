"""Per-frame collision-avoidance pipeline (no GUI, no I/O).

frame -> PWC-Net flow -> ego-motion compensation -> YOLOv9t detection
      -> Kalman/Hungarian tracking -> per-object TTC -> ROI collision check
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import ttc as ttc_heuristics
from .config import PipelineConfig, TtcConfig
from .detection import Detector
from .ego_motion import EgoMotionEstimator, compensate
from .flow import FlowEstimator, resolve_device
from .tracking import Tracker

STAGES = ("flow", "ego_motion", "detection", "tracking", "ttc")


@dataclass
class ObjectState:
    track_id: int
    bbox: np.ndarray
    class_name: str
    lost: int
    in_roi: bool
    ttc_s: float | None
    ttc_ahead_s: float | None  # predicted TTC at the estimator's furthest horizon, if it forecasts
    warning: bool


@dataclass
class FrameResult:
    index: int
    frame: np.ndarray  # resized BGR frame the results refer to
    flow: np.ndarray
    ego_params: np.ndarray | None
    objects: list[ObjectState]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def warnings(self) -> list[ObjectState]:
        return [o for o in self.objects if o.warning]


@contextmanager
def _timed(timings_ms: dict[str, float], stage: str):
    t0 = time.perf_counter()
    yield
    timings_ms[stage] = (time.perf_counter() - t0) * 1000.0


class HeuristicTtcEstimator:
    """Default TTC source: the legacy divergence/flow/looming fusion (no forecast)."""

    def __init__(self, cfg: TtcConfig):
        self.cfg = cfg

    def __call__(self, flow, div_map, tracks, fps) -> dict[int, tuple[float | None, float | None]]:
        return {tid: (ttc_heuristics.estimate_ttc(flow, div_map, t.bbox, fps, self.cfg), None) for tid, t in tracks.items()}


class CollisionPipeline:
    def __init__(self, cfg: PipelineConfig, fps: float, ttc_estimator=None):
        self.cfg = cfg
        self.fps = fps
        self.ttc_estimator = ttc_estimator or HeuristicTtcEstimator(cfg.ttc)
        device = resolve_device(cfg.device)
        self.flow = FlowEstimator(cfg.flow_model, cfg.flow_checkpoint, (cfg.frame_height, cfg.frame_width), device, cfg.flow_backend)
        self.detector = Detector(cfg.yolo_weights, cfg.conf_threshold, device)
        self.tracker = Tracker(cfg.iou_threshold, cfg.max_track_lost)
        self.ego = EgoMotionEstimator(cfg.ego)
        self._prev: np.ndarray | None = None
        self._index = 0

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        self._index += 1
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        frame = cv2.resize(frame_bgr, self.cfg.frame_size)
        h, w = frame.shape[:2]

        with _timed(timings, "flow"):
            flow = self.flow(self._prev, frame) if self._prev is not None else np.zeros((h, w, 2), np.float32)

        with _timed(timings, "ego_motion"):
            ego_params = None
            if self._prev is not None:
                prev_gray = cv2.cvtColor(self._prev, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ego_params = self.ego.update(self._index, prev_gray, curr_gray)
                flow = compensate(flow, ego_params)

        with _timed(timings, "detection"):
            detections = self.detector(frame)

        with _timed(timings, "tracking"):
            tracks = self.tracker.update(detections, flow)

        with _timed(timings, "ttc"):
            objects = self._assess(flow, tracks) if self._prev is not None else []

        self._prev = frame
        timings["total"] = (time.perf_counter() - t_start) * 1000.0
        return FrameResult(self._index, frame, flow, ego_params, objects, timings)

    def _assess(self, flow: np.ndarray, tracks) -> list[ObjectState]:
        estimates = self.ttc_estimator(flow, ttc_heuristics.divergence(flow), tracks, self.fps)
        ttc_cfg = self.cfg.ttc
        objects = []
        for tid, track in tracks.items():
            in_roi = self.cfg.roi.contains_box_center(track.bbox)
            ttc_s, ttc_ahead_s = estimates.get(tid, (None, None))
            threshold = ttc_cfg.threshold_inside_roi_s if in_roi else ttc_cfg.threshold_outside_roi_s
            warning = ttc_s is not None and ttc_s <= threshold
            objects.append(ObjectState(tid, track.bbox, track.class_name, track.lost, in_roi, ttc_s, ttc_ahead_s, warning))
        return objects
