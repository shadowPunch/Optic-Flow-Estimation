"""ESN-based TTC estimator pluggable into CollisionPipeline (`--ttc-model`)."""

from collision_avoidance.config import TtcConfig

from .features import object_features
from .model import EsnTtcModel, OnlineTtcPredictor


class EsnTtcEstimator:
    """Streams per-track features through the ESN; returns (TTC now, TTC at the last horizon)."""

    def __init__(self, model_path: str, cfg: TtcConfig):
        self.cfg = cfg
        self.predictor = OnlineTtcPredictor(EsnTtcModel.load(model_path))
        self.prev_boxes = {}

    def __call__(self, flow, div_map, tracks, fps) -> dict[int, tuple[float | None, float | None]]:
        estimates = {}
        for tid, track in tracks.items():
            feats = object_features(flow, div_map, track.bbox, self.prev_boxes.get(tid), fps, self.cfg)
            if feats is None:
                estimates[tid] = (None, None)
                continue
            ttc = self.predictor.update(tid, feats)
            estimates[tid] = (float(ttc[0]), float(ttc[-1]))
        self.prev_boxes = {tid: t.bbox for tid, t in tracks.items()}
        self.predictor.forget_missing(tracks.keys())
        return estimates
