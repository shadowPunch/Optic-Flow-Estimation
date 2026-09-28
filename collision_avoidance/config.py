"""Pipeline configuration. Defaults reproduce legacy/ttc-video_feed.py."""

from dataclasses import dataclass, field

# COCO class ids considered collision-relevant.
RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
}


@dataclass(frozen=True)
class Roi:
    """Critical region in front of the vehicle, in resized-frame pixels."""

    left: int = 140
    top: int = 100
    right: int = 360
    bottom: int = 300

    def contains_box_center(self, bbox) -> bool:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return self.left <= cx <= self.right and self.top <= cy <= self.bottom


@dataclass(frozen=True)
class EgoMotionConfig:
    enabled: bool = True
    update_interval: int = 15  # re-estimate every N frames, reuse in between
    num_candidates: int = 32
    max_iterations: int = 15
    mse_threshold: float = 1e-2
    mutation_std: float = 0.32
    translation_range_px: float = 3.0
    # Legacy sampled theta from the same +-3 range as translation (+-172 deg);
    # the HLS port narrowed it, and so do we.
    rotation_range_rad: float = 0.05
    seed: int | None = 0


@dataclass(frozen=True)
class TtcConfig:
    focal_length_px: float = 500.0
    object_size_m: float = 1.5
    min_valid_s: float = 0.1
    max_valid_s: float = 30.0
    # Fusion weights for the divergence, flow-magnitude and looming estimates.
    weight_divergence: float = 3.0
    weight_flow: float = 2.0
    weight_looming: float = 1.0
    threshold_inside_roi_s: float = 1.12
    threshold_outside_roi_s: float = 0.56


@dataclass(frozen=True)
class PipelineConfig:
    yolo_weights: str = "yolov9t.pt"
    flow_model: str = "pwcnet"  # ptlflow name; 'pwcnet' is PWC-DC-Net
    flow_checkpoint: str = "things"
    frame_width: int = 512
    frame_height: int = 384
    device: str = "auto"
    conf_threshold: float = 0.5
    iou_threshold: float = 0.3
    max_track_lost: int = 30
    roi: Roi = field(default_factory=Roi)
    ego: EgoMotionConfig = field(default_factory=EgoMotionConfig)
    ttc: TtcConfig = field(default_factory=TtcConfig)

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) as expected by cv2.resize."""
        return self.frame_width, self.frame_height
