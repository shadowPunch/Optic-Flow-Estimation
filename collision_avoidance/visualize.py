"""Drawing helpers for the two display windows (tracking view and flow/TTC view)."""

import cv2
import numpy as np

from .config import PipelineConfig
from .flow import flow_to_bgr
from .pipeline import FrameResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, ORANGE, YELLOW, GREEN, WHITE, CYAN = (0, 0, 255), (0, 100, 255), (0, 255, 255), (0, 255, 0), (255, 255, 255), (255, 255, 0)


def draw_roi(image: np.ndarray, cfg: PipelineConfig) -> None:
    roi, ttc = cfg.roi, cfg.ttc
    cv2.rectangle(image, (roi.left, roi.top), (roi.right, roi.bottom), CYAN, 2)
    cv2.putText(image, f"ROI ({ttc.threshold_inside_roi_s}s)", (roi.left, roi.top - 10), FONT, 0.6, CYAN, 2)
    cv2.putText(image, f"Outside ROI: {ttc.threshold_outside_roi_s}s", (10, 30), FONT, 0.5, WHITE, 1)
    cv2.putText(image, f"Inside ROI: {ttc.threshold_inside_roi_s}s", (10, 50), FONT, 0.5, CYAN, 1)


def tracking_view(result: FrameResult, cfg: PipelineConfig) -> np.ndarray:
    image = result.frame.copy()
    draw_roi(image, cfg)
    if result.ego_params is not None:
        dx, dy, theta = result.ego_params
        text = f"Ego Motion: dx={dx:.1f}, dy={dy:.1f}, yaw={np.degrees(theta):.1f}deg"
        cv2.putText(image, text, (10, 110), FONT, 0.4, WHITE, 1)
    for obj in result.objects:
        x1, y1, x2, y2 = map(int, obj.bbox)
        color = YELLOW if obj.in_roi else GREEN
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"ID:{obj.track_id} {obj.class_name}", (x1, y1 - 5), FONT, 0.4, color, 1)
    cv2.putText(image, f"Tracks: {len(result.objects)}", (10, image.shape[0] - 20), FONT, 0.5, WHITE, 1)
    return image


def flow_view(result: FrameResult, cfg: PipelineConfig) -> np.ndarray:
    image = flow_to_bgr(result.flow)
    draw_roi(image, cfg)
    total_ms = result.timings_ms.get("total", 0.0)
    cv2.putText(image, f"Processing: {total_ms:.1f}ms", (10, 70), FONT, 0.5, WHITE, 1)
    cv2.putText(image, f"FPS: {1000 / max(total_ms, 1):.1f}", (10, 90), FONT, 0.5, WHITE, 1)

    for obj in result.warnings:
        x1, y1, x2, y2 = map(int, obj.bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color, thickness = (RED, 4) if obj.in_roi else (ORANGE, 3)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        ahead = f" -> {obj.ttc_ahead_s:.1f}s" if obj.ttc_ahead_s is not None else ""  # forecast
        label = f"COLLISION! {obj.class_name} TTC: {obj.ttc_s:.1f}s{ahead} ({'ROI' if obj.in_roi else 'OUT'})"
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 2)
        cv2.rectangle(image, (cx - tw // 2 - 5, cy - 20), (cx + tw // 2 + 5, cy), (0, 0, 0), -1)
        cv2.putText(image, label, (cx - tw // 2, cy - 5), FONT, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(image, f"ID:{obj.track_id}", (x1, y1 - 5), FONT, 0.4, color, 1, cv2.LINE_AA)

    warnings = result.warnings
    if warnings:
        bottom = image.shape[0]
        cv2.putText(image, f"COLLISION WARNINGS: {len(warnings)}", (10, bottom - 50), FONT, 0.7, RED, 2, cv2.LINE_AA)
        for i, obj in enumerate(warnings[:3]):
            text = f"ID{obj.track_id}: {obj.class_name} {obj.ttc_s:.1f}s"
            cv2.putText(image, text, (10, bottom - 30 + i * 15), FONT, 0.4, RED, 1, cv2.LINE_AA)
    return image
