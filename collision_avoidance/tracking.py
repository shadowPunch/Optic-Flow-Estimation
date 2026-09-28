"""Multi-object tracking: constant-velocity Kalman filter + Hungarian IoU matching.

Tracks that miss a detection are propagated by the Kalman prediction nudged by
the mean optical flow around the predicted centre, so objects survive short
detector dropouts.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    bbox: np.ndarray  # (x1, y1, x2, y2) int
    class_name: str
    confidence: float


@dataclass
class Track:
    kf: cv2.KalmanFilter
    bbox: np.ndarray
    class_name: str
    lost: int = 0
    age: int = 1


def iou(a, b) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter + 1e-6)


def _box_center(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def _measurement(x: float, y: float) -> np.ndarray:
    return np.array([[np.float32(x)], [np.float32(y)]])


def create_kalman_filter(cx: float, cy: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)  # state (x, y, vx, vy), measurement (x, y)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    kf.statePost = kf.statePre.copy()
    return kf


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30, flow_window: int = 10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.flow_window = flow_window
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, detections: list[Detection], flow: np.ndarray) -> dict[int, Track]:
        matched, unmatched_tracks, unmatched_dets = self._associate(detections)
        for tid, det_idx in matched:
            self._update_matched(self.tracks[tid], detections[det_idx])
        for tid in unmatched_tracks:
            self._propagate_with_flow(self.tracks[tid], flow)
        for det_idx in unmatched_dets:
            self._create(detections[det_idx])
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.lost <= self.max_lost}
        return self.tracks

    def _associate(self, detections):
        track_ids = list(self.tracks)
        unmatched_tracks, unmatched_dets = list(track_ids), list(range(len(detections)))
        matched = []
        if not detections or not track_ids:
            return matched, unmatched_tracks, unmatched_dets
        ious = np.array([[iou(self.tracks[tid].bbox, d.bbox) for d in detections] for tid in track_ids])
        for r, c in zip(*linear_sum_assignment(-ious)):
            if ious[r, c] >= self.iou_threshold:
                matched.append((track_ids[r], c))
                unmatched_tracks.remove(track_ids[r])
                unmatched_dets.remove(c)
        return matched, unmatched_tracks, unmatched_dets

    @staticmethod
    def _update_matched(track: Track, det: Detection):
        track.kf.correct(_measurement(*_box_center(det.bbox)))
        track.bbox = det.bbox
        track.class_name = det.class_name
        track.lost = 0
        track.age += 1

    def _propagate_with_flow(self, track: Track, flow: np.ndarray):
        h, w = flow.shape[:2]
        pred = track.kf.predict()
        px, py = int(pred[0, 0]), int(pred[1, 0])
        x1, x2 = max(0, px - self.flow_window), min(w, px + self.flow_window)
        y1, y2 = max(0, py - self.flow_window), min(h, py + self.flow_window)
        if x2 > x1 and y2 > y1:
            mean_flow = flow[y1:y2, x1:x2].mean(axis=(0, 1))
            px += int(mean_flow[0])
            py += int(mean_flow[1])
        track.kf.correct(_measurement(px, py))
        track.lost += 1

    def _create(self, det: Detection):
        self.tracks[self._next_id] = Track(create_kalman_filter(*_box_center(det.bbox)), det.bbox, det.class_name)
        self._next_id += 1
