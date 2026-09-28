import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils

# ---------- CONFIG ---------- #
YOLO_MODEL_PATH = 'yolov8n.pt'
CAM_INDEX = 0
FRAME_WIDTH = 512
FRAME_HEIGHT = 384
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IOU_THRESHOLD = 0.3
CONF_THRESHOLD = 0.5
MAX_TRACK_LOST = 30
DISPLAY_TRACK_AGE = True

# ---------- INIT MODELS ---------- #
yolo = YOLO(YOLO_MODEL_PATH)
pwcnet = ptlflow.get_model('pwcnet', ckpt_path='things').to(DEVICE).eval()

# ---------- UTILITIES ---------- #
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_flow(prev_frame, curr_frame):
    resized_prev = cv2.resize(prev_frame, (FRAME_WIDTH, FRAME_HEIGHT))
    resized_curr = cv2.resize(curr_frame, (FRAME_WIDTH, FRAME_HEIGHT))

    io_adapter = IOAdapter(pwcnet, (FRAME_HEIGHT, FRAME_WIDTH))
    inputs = io_adapter.prepare_inputs([resized_prev, resized_curr])
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        predictions = pwcnet(inputs)

    flow = predictions['flows'][0, 0].permute(1, 2, 0).cpu().numpy()
    return flow

def draw_flow_only(flow):
    flow_rgb = flow_utils.flow_to_rgb(flow)
    flow_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
    return flow_bgr

def create_kalman_filter(cx, cy):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    kf.statePost = kf.statePre.copy()
    return kf

# ---------- TRACKING ---------- #
next_id = 0
tracks = {}

def associate_detections(detections):
    matched, unmatched_tracks, unmatched_detections = [], list(tracks.keys()), list(range(len(detections)))
    if detections and tracks:
        iou_matrix = np.zeros((len(tracks), len(detections)))
        track_ids = list(tracks.keys())
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = iou(tracks[tid]['bbox'], det)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= IOU_THRESHOLD:
                tid = track_ids[r]
                matched.append((tid, c))
                unmatched_tracks.remove(tid)
                unmatched_detections.remove(c)
    return matched, unmatched_tracks, unmatched_detections

def update_matched_tracks(matched, detections):
    for tid, det_idx in matched:
        x1, y1, x2, y2 = detections[det_idx]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        tracks[tid]['kf'].correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        tracks[tid]['bbox'] = detections[det_idx]
        tracks[tid]['lost'] = 0
        tracks[tid]['age'] += 1

def update_unmatched_tracks(unmatched_tracks, flow):
    h, w = flow.shape[:2]
    for tid in unmatched_tracks:
        pred = tracks[tid]['kf'].predict()
        px, py = int(pred[0]), int(pred[1])
        x1, x2 = max(0, px - 5), min(w, px + 5)
        y1, y2 = max(0, py - 5), min(h, py + 5)
        if x2 > x1 and y2 > y1:
            avg_flow = flow[y1:y2, x1:x2].mean(axis=(0, 1))
            px += int(avg_flow[0])
            py += int(avg_flow[1])
        tracks[tid]['kf'].correct(np.array([[np.float32(px)], [np.float32(py)]]))
        tracks[tid]['lost'] += 1

def create_new_tracks(unmatched_detections, detections):
    global next_id
    for det_idx in unmatched_detections:
        x1, y1, x2, y2 = detections[det_idx]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        kf = create_kalman_filter(cx, cy)
        tracks[next_id] = {'kf': kf, 'bbox': detections[det_idx], 'lost': 0, 'age': 1}
        next_id += 1

def remove_lost_tracks():
    remove_ids = [tid for tid in tracks if tracks[tid]['lost'] > MAX_TRACK_LOST]
    for tid in remove_ids:
        del tracks[tid]

def draw_tracks(display_frame):
    for tid, data in tracks.items():
        pred = data['kf'].predict()
        px, py = int(pred[0]), int(pred[1])
        x1, y1, x2, y2 = data['bbox']
        color = (0, 255, 0) if data['lost'] == 0 else (0, 128, 255)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid}"
        if DISPLAY_TRACK_AGE:
            label += f" Age:{data['age']}"
        cv2.putText(display_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.circle(display_frame, (px, py), 4, (0, 0, 255), -1)

# ---------- MAIN LOOP ---------- #
cap = cv2.VideoCapture(CAM_INDEX)
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    if prev_frame is not None:
        flow = compute_flow(prev_frame, frame)
    else:
        flow = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2), dtype=np.float32)

    flow_only = draw_flow_only(flow)
    tracking_frame = frame_resized.copy()

    # Detection
    results = yolo(frame, verbose=False)[0]
    if results.boxes is not None:
        for box in results.boxes:
            if box.conf.cpu().item() >= CONF_THRESHOLD:
                detections.append(box.xyxy.cpu().numpy().astype(int).flatten())

    # Tracking updates
    matched, unmatched_tracks, unmatched_detections = associate_detections(detections)
    update_matched_tracks(matched, detections)
    update_unmatched_tracks(unmatched_tracks, flow)
    create_new_tracks(unmatched_detections, detections)
    remove_lost_tracks()
    draw_tracks(tracking_frame)

    prev_frame = frame.copy()

    cv2.imshow("Tracking View", tracking_frame)
    cv2.imshow("Optical Flow View", flow_only)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
