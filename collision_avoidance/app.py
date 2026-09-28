"""Command-line entry point: run the pipeline on a video file or webcam.

    collision-avoidance --source data/videos/00049.mp4
    collision-avoidance --source 0                      # webcam
    collision-avoidance --source clip.mp4 --no-display --max-frames 300   # headless benchmark

Keys (display mode): ESC quit, P pause, R toggle real-time pacing,
E toggle ego-motion correction, S save frames, H help.
"""

import argparse
import dataclasses
import time

import cv2
import numpy as np

from .config import PipelineConfig
from .pipeline import STAGES, CollisionPipeline, FrameResult
from .telemetry import start_run
from .visualize import flow_view, tracking_view

HELP = "ESC: exit | P: pause | R: real-time pacing | E: ego-motion | S: save frames | H: help"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path or webcam index")
    p.add_argument("--yolo-weights", default=PipelineConfig.yolo_weights)
    p.add_argument("--device", default="auto")
    p.add_argument("--ttc-model", help="trained ESN checkpoint (ttc_esn.train); default: legacy heuristic")
    p.add_argument("--no-ego", action="store_true", help="disable ego-motion compensation")
    p.add_argument("--no-display", action="store_true", help="headless (benchmarking / servers)")
    p.add_argument("--no-realtime", action="store_true", help="do not pace playback to the source FPS")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--output", help="write side-by-side annotated video to this path")
    return p.parse_args(argv)


def open_source(source: str) -> tuple[cv2.VideoCapture, float, str]:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        kind = "webcam"
    else:
        cap = cv2.VideoCapture(source)
        kind = "video"
    if not cap.isOpened():
        raise SystemExit(f"Could not open {kind} source '{source}'")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, fps, kind


def latency_summary(timings: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for stage in (*STAGES, "total"):
        values = np.array([t[stage] for t in timings if stage in t])
        if values.size:
            summary[f"latency_ms/{stage}_mean"] = float(values.mean())
            summary[f"latency_ms/{stage}_p50"] = float(np.percentile(values, 50))
            summary[f"latency_ms/{stage}_p95"] = float(np.percentile(values, 95))
    return summary


def handle_key(key: int, pipeline: CollisionPipeline, state: dict, views) -> bool:
    """Apply a keyboard command; return False to quit."""
    if key == 27:
        return False
    if key == ord("p"):
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)
    elif key == ord("r"):
        state["realtime"] = not state["realtime"]
        print(f"Real-time playback: {state['realtime']}")
    elif key == ord("e"):
        ego = pipeline.ego
        ego.cfg = dataclasses.replace(ego.cfg, enabled=not ego.cfg.enabled)
        print(f"Ego-motion correction: {ego.cfg.enabled}")
    elif key == ord("s"):
        stamp = int(time.time())
        cv2.imwrite(f"frame_{stamp}_original.jpg", views[0])
        cv2.imwrite(f"frame_{stamp}_flow.jpg", views[1])
        print(f"Saved frames at timestamp {stamp}")
    elif key == ord("h"):
        print(HELP)
    return True


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = PipelineConfig(yolo_weights=args.yolo_weights, device=args.device)
    if args.no_ego:
        cfg = dataclasses.replace(cfg, ego=dataclasses.replace(cfg.ego, enabled=False))

    cap, fps, kind = open_source(args.source)
    run = start_run(
        "inference",
        {**dataclasses.asdict(cfg), "source": args.source, "source_kind": kind, "fps": fps,
         "max_frames": args.max_frames, "ttc_model": args.ttc_model},
        tags=["pipeline", kind, "host", "esn" if args.ttc_model else "heuristic"],
    )
    ttc_estimator = None
    if args.ttc_model:
        from ttc_esn.online import EsnTtcEstimator
        ttc_estimator = EsnTtcEstimator(args.ttc_model, cfg.ttc)
    pipeline = CollisionPipeline(cfg, fps, ttc_estimator)
    writer = None
    state = {"realtime": not args.no_realtime and not args.no_display}
    timings, n_warnings = [], 0
    print(f"Source: {args.source} ({kind}, {fps:.1f} FPS) | ego-motion: {cfg.ego.enabled} | {HELP}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and len(timings) >= args.max_frames):
                break
            t_frame = time.perf_counter()
            result: FrameResult = pipeline.process(frame)
            timings.append(result.timings_ms)
            n_warnings += len(result.warnings)
            if run is not None:
                run.log({**{f"frame_ms/{k}": v for k, v in result.timings_ms.items()},
                         "tracks": len(result.objects), "warnings": len(result.warnings)}, step=result.index)

            if args.no_display and not args.output:
                continue
            views = (tracking_view(result, cfg), flow_view(result, cfg))
            if args.output:
                if writer is None:
                    h, w = views[0].shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
                writer.write(np.hstack(views))
            if args.no_display:
                continue
            cv2.imshow("Tracking (YOLOv9t)", views[0])
            cv2.imshow("Ego-motion corrected flow + TTC", views[1])
            wait_ms = 1
            if state["realtime"]:
                wait_ms = max(1, int((1.0 / fps - (time.perf_counter() - t_frame)) * 1000))
            if not handle_key(cv2.waitKey(wait_ms) & 0xFF, pipeline, state, views):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    # First frame has no flow; exclude it from latency statistics.
    summary = {**latency_summary(timings[1:]), "frames": len(timings), "warnings_total": n_warnings}
    for key, value in summary.items():
        print(f"{key:32s} {value:10.2f}")
    if run is not None:
        run.summary.update(summary)
        run.finish()
