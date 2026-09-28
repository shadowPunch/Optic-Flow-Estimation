"""Build the ESN dataset: run the deployed pipeline over EvTTC and record target features.

For every sequence:
  1. estimate the video/GT time offset (evttc.estimate_sync_offset),
  2. run CollisionPipeline (PWC-Net flow, ego-motion, YOLOv9t, tracker) on the
     left RGB camera from `warmup` frames before the GT window to its end,
  3. identify the target's track: best IoU with the annotated box, extended to
     unannotated frames by track-id continuity,
  4. store the 17 features of the target per frame plus GT TTC at each horizon.

    python -m ttc_esn.extract --data data/evttc --out outputs/features
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from collision_avoidance.config import PipelineConfig
from collision_avoidance.pipeline import CollisionPipeline
from collision_avoidance.telemetry import start_run
from collision_avoidance.tracking import iou
from collision_avoidance.ttc import divergence

from . import evttc
from .features import FEATURE_NAMES, NUM_FEATURES, object_features

HORIZONS_S = (0.0, 0.1, 0.2, 0.3)
PANEL_W, PANEL_H = 1920, 1200
MIN_TARGET_IOU = 0.3


def _run_pipeline(seq_dir: Path, cfg: PipelineConfig, fps: float, start: int, last: int) -> dict[int, dict]:
    """frame -> {track_id: (bbox, features, heuristic_ttc)} for frames start..last."""
    pipeline = CollisionPipeline(cfg, fps)
    per_frame, prev_boxes = {}, {}
    for index, panel in evttc.iter_left_frames(seq_dir / "video.mp4"):
        if index < start:
            continue
        if index > last:
            break
        result = pipeline.process(panel)
        div_map = divergence(result.flow)
        entries = {}
        for obj in result.objects:
            feats = object_features(result.flow, div_map, obj.bbox, prev_boxes.get(obj.track_id), fps, cfg.ttc)
            if feats is not None:
                entries[obj.track_id] = (obj.bbox, feats, obj.ttc_s)
        prev_boxes = {o.track_id: o.bbox for o in result.objects}
        per_frame[index] = entries
    return per_frame


def _target_tracks(per_frame: dict[int, dict], annotations: dict[int, np.ndarray], scale: np.ndarray) -> dict[int, int]:
    target = {}
    for frame, box in annotations.items():
        candidates = per_frame.get(frame, {})
        scored = [(iou(bbox, box * scale), tid) for tid, (bbox, _, _) in candidates.items()]
        if scored and max(scored)[0] >= MIN_TARGET_IOU:
            target[frame] = max(scored)[1]
    anchors = np.array(sorted(target))
    if anchors.size == 0:
        return target
    for frame, candidates in per_frame.items():
        if frame not in target:
            tid = target[int(anchors[np.abs(anchors - frame).argmin()])]
            if tid in candidates:
                target[frame] = tid
    return target


def extract_sequence(seq_dir: Path, cfg: PipelineConfig, warmup: int) -> dict:
    gt, annotations = evttc.load_gt(seq_dir), evttc.load_annotations(seq_dir)
    fps = evttc.video_fps(seq_dir / "video.mp4")
    sync = evttc.estimate_sync_offset(annotations, gt, fps)
    first = math.ceil((gt.t.iloc[0] - sync.offset_s) * fps)
    last = math.floor((gt.t.iloc[-1] - sync.offset_s) * fps)
    start = max(0, first - warmup)

    per_frame = _run_pipeline(seq_dir, cfg, fps, start, last)
    scale = np.array([cfg.frame_width / PANEL_W, cfg.frame_height / PANEL_H] * 2)
    target = _target_tracks(per_frame, annotations, scale)

    frames = np.arange(start, last + 1)
    times = frames / fps + sync.offset_s
    features = np.full((len(frames), NUM_FEATURES), np.nan, np.float32)
    heuristic_ttc = np.full(len(frames), np.nan, np.float32)
    track_ids = np.full(len(frames), -1, np.int64)
    for k, frame in enumerate(frames):
        tid = target.get(int(frame))
        if tid is not None:
            _, feats, ttc_s = per_frame[int(frame)][tid]
            features[k], track_ids[k] = feats, tid
            heuristic_ttc[k] = np.nan if ttc_s is None else ttc_s

    labels = np.full((len(frames), len(HORIZONS_S)), np.nan, np.float32)
    for j, h in enumerate(HORIZONS_S):
        inside = (times + h >= gt.t.iloc[0]) & (times + h <= gt.t.iloc[-1])
        labels[inside, j] = np.interp(times[inside] + h, gt.t, gt.ttc)

    return {
        "frames": frames, "times": times, "features": features, "labels": labels,
        "heuristic_ttc": heuristic_ttc, "track_ids": track_ids, "fps": fps,
        "sync_offset_s": sync.offset_s, "sync_log_residual_std": sync.log_residual_std,
        "sync_implied_width_m": sync.implied_width_m,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/evttc"))
    p.add_argument("--out", type=Path, default=Path("outputs/features"))
    p.add_argument("--warmup", type=int, default=20, help="frames processed before the GT window")
    p.add_argument("--sequences", nargs="*", help="subset of sequence folder names")
    args = p.parse_args(argv)

    cfg = PipelineConfig()
    seq_dirs = sorted(d for d in args.data.iterdir() if (d / "video.mp4").exists() and (d / "gt_ttc.csv").exists())
    if args.sequences:
        seq_dirs = [d for d in seq_dirs if d.name in args.sequences]
    args.out.mkdir(parents=True, exist_ok=True)
    run = start_run("feature-extraction", {**asdict(cfg), "warmup": args.warmup, "horizons_s": HORIZONS_S,
                                           "features": FEATURE_NAMES, "sequences": [d.name for d in seq_dirs]},
                    tags=["esn", "dataset", "evttc"])
    rows = []
    for seq_dir in seq_dirs:
        data = extract_sequence(seq_dir, cfg, args.warmup)
        np.savez_compressed(args.out / f"{seq_dir.name}.npz", **data, horizons_s=np.array(HORIZONS_S), feature_names=np.array(FEATURE_NAMES))
        in_gt = ~np.isnan(data["labels"][:, 0])
        coverage = float((~np.isnan(data["features"][in_gt, 0])).mean())
        both = in_gt & ~np.isnan(data["heuristic_ttc"])
        rel_err = np.abs(data["heuristic_ttc"][both] - data["labels"][both, 0]) / data["labels"][both, 0]
        row = [seq_dir.name, int(in_gt.sum()), coverage, data["sync_offset_s"], data["sync_log_residual_std"],
               data["sync_implied_width_m"], float(np.median(rel_err)) if rel_err.size else float("nan")]
        rows.append(row)
        print("{:22s} gt_frames={:4d} target_coverage={:.2f} sync={:+.3f}s resid={:.4f} width={:.2f}m heuristic_median_rel_err={:.2f}".format(*row))
    if run is not None:
        import wandb
        run.log({"sequences": wandb.Table(columns=["sequence", "gt_frames", "target_coverage", "sync_offset_s",
                                                   "sync_log_residual_std", "implied_width_m", "heuristic_median_rel_err"], data=rows)})
        run.finish()


if __name__ == "__main__":
    main()
