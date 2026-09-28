import numpy as np
import pytest

from collision_avoidance.config import TtcConfig
from collision_avoidance.ttc import divergence
from ttc_esn.features import FEATURE_NAMES, NUM_FEATURES, object_features
from ttc_esn.model import EsnConfig, EsnTtcModel, OnlineTtcPredictor
from ttc_esn.train import segments

SMALL = EsnConfig(reservoir_size=40, hidden=8, dropout=0.0)


def looming_flow(rate, center=(256, 192), h=384, w=512):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([(xs - center[0]) * rate, (ys - center[1]) * rate], axis=-1)


def test_feature_vector_shape_and_expansion_rate():
    fps, rate = 20.0, 0.01  # 1 % growth per frame -> 1/TTC = 0.2 /s
    flow = looming_flow(rate) + np.array([3.0, -1.0], np.float32)  # plus object translation
    feats = object_features(flow, divergence(flow), [200, 150, 312, 234], [201, 151, 311, 233], fps, TtcConfig())
    assert feats.shape == (NUM_FEATURES,) and np.isfinite(feats).all()
    assert feats[FEATURE_NAMES.index("expansion_rate")] == pytest.approx(rate * fps, rel=1e-4)
    assert feats[FEATURE_NAMES.index("u_mean")] == pytest.approx((3.0 + rate * (255.5 - 256)) * fps, rel=1e-3)
    assert feats[FEATURE_NAMES.index("scale_rate")] > 0


def test_box_outside_frame_gives_none():
    flow = looming_flow(0.01)
    assert object_features(flow, divergence(flow), [600, 10, 700, 50], None, 20.0, TtcConfig()) is None


def test_segments_split_on_gaps_and_track_changes():
    feats = np.ones((8, NUM_FEATURES), np.float32)
    feats[3] = np.nan
    seq = {"features": feats, "track_ids": np.array([1, 1, 1, -1, 1, 1, 2, 2])}
    assert segments(seq) == [slice(0, 3), slice(4, 6), slice(6, 8)]


@pytest.mark.parametrize("use_reservoir", [True, False])
def test_online_predictor_matches_batch_and_save_roundtrip(tmp_path, use_reservoir):
    rng = np.random.default_rng(0)
    model = EsnTtcModel(NUM_FEATURES, (0.0, 0.3), SMALL, use_reservoir)
    feats = rng.normal(size=(12, NUM_FEATURES)).astype(np.float32)
    model.fit_scaler(feats)
    batch = np.exp(model.predict_log_ttc(model.design_matrix(feats)))

    online = OnlineTtcPredictor(model)
    streamed = np.stack([online.update(7, f) for f in feats])
    np.testing.assert_allclose(streamed, batch, rtol=1e-5)

    model.save(tmp_path / "m.pt")
    restored = EsnTtcModel.load(tmp_path / "m.pt")
    np.testing.assert_allclose(np.exp(restored.predict_log_ttc(restored.design_matrix(feats))), batch, rtol=1e-6)
