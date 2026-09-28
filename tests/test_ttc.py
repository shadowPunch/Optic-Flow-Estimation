import numpy as np
import pytest

from collision_avoidance import ttc
from collision_avoidance.config import TtcConfig
from legacy_loader import load_legacy

FPS = 30
CFG = TtcConfig()


@pytest.fixture(scope="module")
def legacy():
    return load_legacy("compute_divergence", "compute_object_ttc_improved", constants=("FOCAL_LENGTH", "OBJECT_SIZE"))


def expanding_flow(h=384, w=512, center=(256, 200), rate=0.02):
    """Flow of a fronto-parallel plane approaching the camera: v = rate * (p - c)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([(xs - center[0]) * rate, (ys - center[1]) * rate], axis=-1)


def random_boxes(rng, n, w=512, h=384):
    for _ in range(n):
        x1, y1 = rng.integers(-20, w - 10), rng.integers(-20, h - 10)
        yield np.array([x1, y1, x1 + rng.integers(4, 200), y1 + rng.integers(4, 200)])


def test_divergence_matches_legacy(legacy):
    flow = np.random.default_rng(0).normal(0, 2, (96, 128, 2)).astype(np.float32)
    np.testing.assert_allclose(ttc.divergence(flow), legacy["compute_divergence"](flow))


@pytest.mark.parametrize("seed", range(5))
def test_fused_ttc_matches_legacy_on_random_flow(legacy, seed):
    rng = np.random.default_rng(seed)
    flow = (expanding_flow(rate=rng.uniform(0.005, 0.05)) + rng.normal(0, 0.5, (384, 512, 2))).astype(np.float32)
    div_map = ttc.divergence(flow)
    for box in random_boxes(rng, 40):
        expected = legacy["compute_object_ttc_improved"](flow, div_map, box, FPS)
        actual = ttc.estimate_ttc(flow, div_map, box, FPS, CFG)
        if expected is None:
            assert actual is None
        else:
            # float32 rounding differs (np.hypot vs sqrt(u^2 + v^2)); behaviour is identical.
            assert actual == pytest.approx(expected, rel=1e-5)


def test_divergence_ttc_on_ideal_looming():
    rate = 0.02  # pixels expand by 2 % per frame -> true TTC = 1 / rate frames
    flow = expanding_flow(rate=rate)
    est = ttc.ttc_from_divergence(ttc.divergence(flow)[150:250, 200:300], FPS)
    # The heuristic omits the factor 2 of 2-D divergence (div = 2 * rate): documented bias.
    assert est == pytest.approx(1.0 / (2 * rate * FPS), rel=1e-6)


def test_invalid_boxes_return_none():
    flow = expanding_flow()
    div_map = ttc.divergence(flow)
    assert ttc.estimate_ttc(flow, div_map, [600, 10, 700, 50], FPS, CFG) is None
    assert ttc.estimate_ttc(np.zeros_like(flow), div_map * 0, [10, 10, 60, 60], FPS, CFG) is None
