"""The 17 per-object optical-flow features fed to the ESN.

All rates are converted to per-second units (flow is px/frame, multiplied by
fps) so a model trained on 20 FPS EvTTC transfers to a 30 FPS camera. Box
geometry is normalised by the frame size. Inverse TTCs (1/s) are used instead
of TTCs because they are bounded and continuous when nothing approaches.
"""

import numpy as np

from collision_avoidance import ttc as heuristics
from collision_avoidance.config import TtcConfig

FEATURE_NAMES = (
    "div_mean", "div_median", "div_std",          # flow divergence inside the box      [1/s]
    "mag_mean", "mag_upper_mean", "mag_std",      # flow magnitude                      [px/s]
    "u_mean", "v_mean",                           # mean flow (object translation)      [px/s]
    "expansion_rate", "expansion_residual",       # least-squares radial expansion fit  [1/s], [px/s]
    "scale_rate",                                 # relative bbox growth                [1/s]
    "box_w", "box_h", "box_cx", "box_cy",         # bbox geometry, fraction of frame
    "inv_ttc_divergence", "inv_ttc_fused",        # heuristic estimates                 [1/s]
)
NUM_FEATURES = len(FEATURE_NAMES)


def _inverse(ttc_s: float | None) -> float:
    return 0.0 if ttc_s is None else 1.0 / ttc_s


def object_features(flow: np.ndarray, div_map: np.ndarray, bbox, prev_bbox, fps: float, cfg: TtcConfig) -> np.ndarray | None:
    """Feature vector for one tracked object, or None if the box lies outside the frame."""
    h, w = div_map.shape
    box = heuristics.clip_box(bbox, w, h)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    u, v = flow[y1:y2, x1:x2, 0], flow[y1:y2, x1:x2, 1]
    div = div_map[y1:y2, x1:x2]
    mag = np.hypot(u, v)
    upper = mag[mag > np.percentile(mag, 50)]

    # Fit flow = t + a * (p - c): a > 0 means the object grows in the image (approaches).
    ys, xs = np.mgrid[y1:y2, x1:x2]
    rx, ry = xs - (x1 + x2 - 1) / 2.0, ys - (y1 + y2 - 1) / 2.0
    denom = float((rx * rx + ry * ry).sum())
    du, dv = u - u.mean(), v - v.mean()
    a = float((du * rx + dv * ry).sum()) / denom if denom > 0 else 0.0
    residual = float(np.sqrt(np.mean((du - a * rx) ** 2 + (dv - a * ry) ** 2)))

    scale_rate = 0.0
    if prev_bbox is not None:
        area, prev_area = (x2 - x1) * (y2 - y1), (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
        if prev_area > 0:
            scale_rate = (np.sqrt(area / prev_area) - 1.0) * fps

    components = heuristics.estimate_components(flow, div_map, bbox, fps, cfg)
    return np.array([
        div.mean() * fps, np.median(div) * fps, div.std() * fps,
        mag.mean() * fps, (upper.mean() if upper.size else 0.0) * fps, mag.std() * fps,
        u.mean() * fps, v.mean() * fps,
        a * fps, residual * fps,
        scale_rate,
        (x2 - x1) / w, (y2 - y1) / h, (x1 + x2) / (2 * w), (y1 + y2) / (2 * h),
        _inverse(components["divergence"]), _inverse(heuristics.fuse(components, cfg)),
    ], dtype=np.float32)
