import cv2
import numpy as np
import pytest

from collision_avoidance import ego_motion
from collision_avoidance.config import EgoMotionConfig


def textured_image(h=192, w=256, seed=0):
    noise = np.random.default_rng(seed).uniform(0, 255, (h // 8, w // 8)).astype(np.float32)
    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 255).astype(np.uint8)


def test_ego_flow_matches_exact_inverse_mapping():
    h, w, params = 120, 160, (1.5, -0.7, 0.01)
    m = np.vstack([ego_motion.affine_matrix(*params, (w, h)), [0, 0, 1]])
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    exact = (np.linalg.inv(m) @ pts)[:2] - pts[:2]
    approx = ego_motion.ego_flow(params, (h, w)).reshape(-1, 2).T
    # Small-angle linearisation: error grows with theta * translation, well below a pixel here.
    assert np.abs(exact - approx).max() < 0.05


def test_compensation_removes_pure_ego_motion():
    params = np.array([2.0, 1.0, 0.004])
    flow = ego_motion.ego_flow(params, (96, 128))
    np.testing.assert_allclose(ego_motion.compensate(flow, params), 0.0, atol=1e-5)


@pytest.mark.parametrize("true", [(2.0, -1.0, 0.0), (-1.5, 0.5, 0.01)])
def test_genetic_search_recovers_camera_motion(true):
    prev = textured_image()
    # curr is prev seen after the camera moved; warp(curr, true) should give prev back.
    curr = ego_motion.warp(prev, -np.array(true))
    cfg = EgoMotionConfig(num_candidates=48, max_iterations=30, seed=1)
    est = ego_motion.estimate(prev, curr, cfg, np.random.default_rng(cfg.seed))
    assert est[:2] == pytest.approx(true[:2], abs=0.5)
    assert est[2] == pytest.approx(true[2], abs=0.01)
