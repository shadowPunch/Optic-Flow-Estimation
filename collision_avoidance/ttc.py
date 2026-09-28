"""Heuristic time-to-collision (TTC) estimation from dense optical flow.

Three per-object estimates are fused with fixed weights (3:2:1):
  * divergence  - expansion of the flow field inside the box,
  * flow        - mean flow magnitude with a pinhole distance prior,
  * looming     - radial flow around the box centre relative to box size.

The formulas intentionally match the original pipeline, whose collision
thresholds were tuned against them. Known simplifications (see README):
the divergence estimate omits the factor 2 of a looming plane, and the
distance prior cancels out of the flow-magnitude estimate.
"""

import cv2
import numpy as np

from .config import TtcConfig

_LOOMING_OFFSETS = ((0, 0), (-5, 0), (5, 0), (0, -5), (0, 5))


def divergence(flow: np.ndarray) -> np.ndarray:
    """du/dx + dv/dy using a normalised 3x3 Sobel kernel."""
    du_dx = cv2.Sobel(flow[..., 0], cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dv_dy = cv2.Sobel(flow[..., 1], cv2.CV_64F, 0, 1, ksize=3) / 8.0
    return du_dx + dv_dy


def clip_box(bbox, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def ttc_from_divergence(div_roi: np.ndarray, fps: float) -> float | None:
    median_div, mean_div = np.median(div_roi), np.mean(div_roi)
    # Prefer the median unless it is much smaller than the mean (sparse expansion).
    div = median_div if abs(median_div) > abs(mean_div) * 0.5 else mean_div
    if div > 1e-5:
        return 1.0 / (div * fps)
    return None


def ttc_from_flow_magnitude(flow_roi: np.ndarray, box_size_px: int, fps: float, cfg: TtcConfig) -> float | None:
    magnitude = np.hypot(flow_roi[..., 0], flow_roi[..., 1])
    upper_half = magnitude[magnitude > np.percentile(magnitude, 50)]
    if upper_half.size == 0:
        return None
    mean_mag = np.mean(upper_half)
    if mean_mag <= 0.1:
        return None
    distance_m = cfg.object_size_m * cfg.focal_length_px / box_size_px
    speed_m_s = mean_mag * distance_m * fps / cfg.focal_length_px
    if speed_m_s <= 0.1:
        return None
    return distance_m / speed_m_s


def ttc_from_looming(flow: np.ndarray, box: tuple[int, int, int, int], fps: float) -> float | None:
    x1, y1, x2, y2 = box
    h, w = flow.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    radial = []
    for dx, dy in _LOOMING_OFFSETS:
        px, py = cx + dx, cy + dy
        if (dx or dy) and 0 <= px < w and 0 <= py < h:
            u, v = flow[py, px]
            radial.append(abs((u * dx + v * dy) / np.hypot(dx, dy)))
    if not radial:
        return None
    mean_radial = np.mean(radial)
    if mean_radial <= 0.1:
        return None
    return np.sqrt((x2 - x1) * (y2 - y1)) / (mean_radial * fps * 2)


def estimate_components(flow: np.ndarray, div_map: np.ndarray, bbox, fps: float, cfg: TtcConfig) -> dict | None:
    """Return the three raw TTC estimates (seconds or None) for one box."""
    box = clip_box(bbox, div_map.shape[1], div_map.shape[0])
    if box is None:
        return None
    x1, y1, x2, y2 = box
    flow_roi = flow[y1:y2, x1:x2]
    return {
        "divergence": ttc_from_divergence(div_map[y1:y2, x1:x2], fps),
        "flow": ttc_from_flow_magnitude(flow_roi, max(x2 - x1, y2 - y1), fps, cfg),
        "looming": ttc_from_looming(flow, box, fps),
    }


def fuse(components: dict, cfg: TtcConfig) -> float | None:
    weights = {"divergence": cfg.weight_divergence, "flow": cfg.weight_flow, "looming": cfg.weight_looming}
    valid = [(v, weights[k]) for k, v in components.items() if v is not None and cfg.min_valid_s < v < cfg.max_valid_s]
    if not valid:
        return None
    values, w = zip(*valid)
    return float(np.average(values, weights=w))


def estimate_ttc(flow: np.ndarray, div_map: np.ndarray, bbox, fps: float, cfg: TtcConfig) -> float | None:
    components = estimate_components(flow, div_map, bbox, fps, cfg)
    return None if components is None else fuse(components, cfg)
