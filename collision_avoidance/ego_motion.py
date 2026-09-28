"""Ego-motion compensation (GENEVO-style genetic search).

A population of rigid 2-D motions (dx, dy, theta) is evolved so that warping
the current frame by the candidate best matches the previous frame (MSE).
The winning motion is converted into the flow field it would induce and
subtracted from the observed optical flow, leaving object-induced motion.

Deviations from the legacy script (both documented in README):
  * theta is sampled from +-rotation_range_rad instead of +-3 rad;
  * the rotational ego-flow has the correct sign for cv2's rotation matrix.
"""

from functools import lru_cache

import cv2
import numpy as np

from .config import EgoMotionConfig


def affine_matrix(dx: float, dy: float, theta_rad: float, size_wh: tuple[int, int]) -> np.ndarray:
    """Rotation about the image centre followed by a translation."""
    w, h = size_wh
    m = cv2.getRotationMatrix2D((w // 2, h // 2), np.degrees(theta_rad), 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    return m


def warp(image: np.ndarray, params) -> np.ndarray:
    h, w = image.shape[:2]
    m = affine_matrix(*params, (w, h))
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def estimate(prev_gray: np.ndarray, curr_gray: np.ndarray, cfg: EgoMotionConfig, rng: np.random.Generator) -> np.ndarray:
    """Return (dx, dy, theta) such that warp(curr, params) ~= prev.

    The search runs on images downscaled by `cfg.downscale` (cost falls with the
    square of the factor); translations are scaled back, rotation is scale-free.
    """
    d = cfg.downscale
    if d > 1:
        size = (prev_gray.shape[1] // d, prev_gray.shape[0] // d)
        prev_gray = cv2.resize(prev_gray, size, interpolation=cv2.INTER_AREA)
        curr_gray = cv2.resize(curr_gray, size, interpolation=cv2.INTER_AREA)
    params = _search(prev_gray, curr_gray, cfg, rng, pixel_scale=1.0 / d)
    params[:2] *= d
    return params


def _search(prev_gray, curr_gray, cfg: EgoMotionConfig, rng: np.random.Generator, pixel_scale: float) -> np.ndarray:
    n = cfg.num_candidates
    t, r = cfg.translation_range_px * pixel_scale, cfg.rotation_range_rad
    candidates = rng.uniform([-t, -t, -r], [t, t, r], size=(n, 3))
    candidates[0] = 0.0  # always consider "no motion"
    # Rotation mutates on its own scale, as in the HLS port.
    mutation_scale = np.array([1.0, 1.0, r / t])

    best_params, best_mse = np.zeros(3), float("inf")
    for _ in range(cfg.max_iterations):
        mses = np.array([_mse(prev_gray, warp(curr_gray, c)) for c in candidates])
        order = np.argsort(mses)
        elite = candidates[order[: n // 2]]
        if mses[order[0]] < best_mse:
            best_mse, best_params = mses[order[0]], candidates[order[0]].copy()
        if best_mse < cfg.mse_threshold:
            break
        mutated = elite + rng.normal(0.0, cfg.mutation_std * pixel_scale, elite.shape) * mutation_scale
        crossover = (elite[::2] + elite[1::2]) / 2 if len(elite) >= 2 else elite
        candidates = np.vstack((elite, mutated, crossover))[:n]
    return best_params


@lru_cache(maxsize=4)
def _centred_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs - w // 2, ys - h // 2


def ego_flow(params, shape_hw: tuple[int, int]) -> np.ndarray:
    """Flow (prev -> curr) induced by the camera motion `params`.

    warp(curr, M) ~= prev means a prev pixel p appears at M^-1 p in curr, so the
    ego flow is M^-1 p - p. Small-angle linearisation around the centre.
    """
    dx, dy, theta = params
    h, w = shape_hw
    rel_x, rel_y = _centred_grid(h, w)
    flow = np.empty((h, w, 2), dtype=np.float32)
    flow[..., 0] = -dx - theta * rel_y
    flow[..., 1] = -dy + theta * rel_x
    return flow


def compensate(flow: np.ndarray, params) -> np.ndarray:
    """Remove the ego-motion component from an observed flow field."""
    if params is None:
        return flow
    return flow - ego_flow(params, flow.shape[:2])


class EgoMotionEstimator:
    """Re-estimates ego-motion every `update_interval` frames and reuses it in between."""

    def __init__(self, cfg: EgoMotionConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.params: np.ndarray | None = None

    def update(self, frame_index: int, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray | None:
        if self.cfg.enabled and frame_index % self.cfg.update_interval == 0:
            self.params = estimate(prev_gray, curr_gray, self.cfg, self.rng)
        return self.params if self.cfg.enabled else None
