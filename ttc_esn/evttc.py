"""EvTTC dataset access: download, ground truth, annotations, frames, time sync.

Per sequence we use three artefacts (see evttc_manifest.json):
  * video.mp4     20 FPS 2x2 mosaic; bottom-left panel is the left RGB camera
  * gt_ttc.csv    100 Hz ground truth: index, t [s], distance [m], velocity [m/s], ttc [s]
  * annotations/  ISAT polygons of the target for a subset of frames (NNNN.json)

Video frame i is taken at t = i / fps + offset. The offset is re-estimated per
sequence from the pinhole constraint bbox_width_px * distance = const
(a rigid target), see `estimate_sync_offset`.
"""

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MANIFEST = Path(__file__).with_name("evttc_manifest.json")
DRIVE_DOWNLOAD = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# Left RGB panel inside the 3852x1932 mosaic (found by locating the black separators).
LEFT_RGB_PANEL = (slice(732, 1932), slice(0, 1920))
LEFT_RGB_FX = 1779.08  # blackflys/left/calib/intrinsics[0] from the EvTTC HDF5


def sequence_names() -> list[str]:
    return list(json.loads(MANIFEST.read_text())["sequences"])


def slug(name: str) -> str:
    return re.sub(r"[^\w-]", "", name)


def _fetch(file_id: str, target: Path) -> None:
    # curl handles Drive's large-file confirmation; gdown is refused for the videos.
    if target.exists():
        return
    tmp = target.with_suffix(target.suffix + ".part")
    subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(tmp), DRIVE_DOWNLOAD.format(id=file_id)], check=True)
    # Drive answers quota/permission problems with an HTML page and HTTP 200.
    if tmp.read_bytes()[:512].lstrip().lower().startswith((b"<!doctype html", b"<html")):
        tmp.unlink()
        raise RuntimeError(f"Google Drive refused {file_id} (quota exceeded or not shared); retry later")
    tmp.rename(target)


def download(name: str, root: Path, workers: int = 8) -> Path:
    """Fetch video, GT and annotations for one sequence (skips files already present)."""
    from concurrent.futures import ThreadPoolExecutor

    import gdown

    entry = json.loads(MANIFEST.read_text())["sequences"][name]
    seq_dir = root / slug(name)
    ann_dir = seq_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    _fetch(entry["gt_ttc"], seq_dir / "gt_ttc.csv")
    _fetch(entry["video"], seq_dir / "video.mp4")
    # gdown downloads folders serially and slowly; list once, fetch files in parallel.
    listing = gdown.download_folder(id=entry["annotations_folder"], skip_download=True, quiet=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda f: _fetch(f.id, ann_dir / Path(f.path).name), listing))
    return seq_dir


def load_gt(seq_dir: Path) -> pd.DataFrame:
    return pd.read_csv(seq_dir / "gt_ttc.csv", sep=r"\s+", header=None, names=["index", "t", "distance", "velocity", "ttc"])


def load_annotations(seq_dir: Path) -> dict[int, np.ndarray]:
    """frame index -> target bbox (x1, y1, x2, y2) in left-panel pixels."""
    boxes = {}
    for path in sorted((seq_dir / "annotations").glob("*.json")):
        objects = json.loads(path.read_text())["objects"]
        if not objects:
            continue
        pts = np.concatenate([np.asarray(o["segmentation"], dtype=np.float64) for o in objects])
        boxes[int(path.stem)] = np.array([*pts.min(0), *pts.max(0)])
    return boxes


def iter_left_frames(video_path: Path):
    """Yield (frame_index, left RGB panel as BGR) for every video frame."""
    cap = cv2.VideoCapture(str(video_path))
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield index, frame[LEFT_RGB_PANEL]
            index += 1
    finally:
        cap.release()


def video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


@dataclass(frozen=True)
class SyncResult:
    offset_s: float
    log_residual_std: float  # spread of log(width * distance); ~0 when in sync
    implied_width_m: float
    n_frames: int


def estimate_sync_offset(annotations: dict[int, np.ndarray], gt: pd.DataFrame, fps: float,
                         search_s: float = 5.0, step_s: float = 0.005) -> SyncResult:
    frames = np.array(sorted(annotations))
    widths = np.array([annotations[i][2] - annotations[i][0] for i in frames])
    best = None
    for offset in np.arange(-search_s, search_s + step_s, step_s):
        t = frames / fps + offset
        inside = (t >= gt.t.iloc[0]) & (t <= gt.t.iloc[-1])
        if inside.sum() < 0.9 * len(frames):
            continue
        log_size = np.log(widths[inside] * np.interp(t[inside], gt.t, gt.distance))
        if best is None or log_size.std() < best.log_residual_std:
            best = SyncResult(float(offset), float(log_size.std()), float(np.exp(log_size.mean()) / LEFT_RGB_FX), int(inside.sum()))
    if best is None:
        raise ValueError("annotations do not overlap the ground-truth time range")
    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the EvTTC subset listed in evttc_manifest.json")
    parser.add_argument("--root", type=Path, default=Path("data/evttc"))
    args = parser.parse_args()
    for seq in sequence_names():
        seq_dir = download(seq, args.root)
        print(f"done {seq}: {len(list((seq_dir / 'annotations').glob('*.json')))} annotations", flush=True)
