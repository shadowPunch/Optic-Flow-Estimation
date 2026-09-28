import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils
import os
import time

# ---------- CONFIG ---------- #
YOLO_MODEL_PATH = 'yolov9t.pt'  # Updated to YOLOv9t
WEBCAM_INDEX = 0
FRAME_WIDTH = 512
FRAME_HEIGHT = 384
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IOU_THRESHOLD = 0.3
CONF_THRESHOLD = 0.5
MAX_TRACK_LOST = 30
DIV_THRESHOLD = 0.1
FOCAL_LENGTH = 500
OBJECT_SIZE = 1.5

# Real-time playback settings
REAL_TIME_PLAYBACK = True
SKIP_FRAMES = 1

# ---------- ROI CONFIG ---------- #
ROI_LEFT = 140
ROI_RIGHT = 360
ROI_TOP = 100
ROI_BOTTOM = 300

# Time thresholds for collision warning
TTC_THRESHOLD_INSIDE_ROI = 1.12
TTC_THRESHOLD_OUTSIDE_ROI = 0.56

# ---------- EGO MOTION CORRECTION CONFIG ---------- #
NUM_CANDIDATES = 64
MAX_ITER = 30
MSE_THRESHOLD = 1e-2
MUTATION_STD = 0.4
EGO_MOTION_ENABLED = True

# ---------- RELEVANT OBJECT CLASSES ---------- #
# COCO classes relevant for collision detection
RELEVANT_CLASSES = {
    0: 'person',           # People
    1: 'bicycle',          # Bicycles
    2: 'car',              # Cars
    3: 'motorcycle',       # Motorcycles
    4: 'airplane',         # Low flying aircraft (rare but possible)
    5: 'bus',              # Buses
    6: 'train',            # Trains
    7: 'truck',            # Trucks
    16: 'bird',            # Large birds that could cause collisions
    17: 'cat',             # Cats on road
    18: 'dog',             # Dogs on road
    19: 'horse',           # Horses
    20: 'sheep',           # Livestock
    21: 'cow',             # Livestock
    22: 'elephant',        # Large animals
    23: 'bear',            # Large wild animals
}

# ---------- INIT MODELS ---------- #
try:
    yolo = YOLO(YOLO_MODEL_PATH)
    print(f"Successfully loaded YOLOv9t model from {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLOv9t: {e}")
    print("Falling back to YOLOv8n...")
    yolo = YOLO('yolov8n.pt')

pwcnet = ptlflow.get_model('pwcnet', ckpt_path='things').to(DEVICE).eval()

print(f"Using device: {DEVICE}")
print(f"Using webcam index: {WEBCAM_INDEX}")
print(f"Ego motion correction: {EGO_MOTION_ENABLED}")
print(f"Tracking {len(RELEVANT_CLASSES)} relevant object classes")

# ---------- EGO MOTION CORRECTION ---------- #
def apply_ego_motion_cuda(image, dx, dy, theta_rad):
    """GPU-accelerated image transformation for ego motion correction"""
    try:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(theta_rad), 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        
        # Use CUDA-accelerated warpAffine if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            result_gpu = cv2.cuda.warpAffine(gpu_img, M, (image.shape[1], image.shape[0]),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return result_gpu.download()
        else:
            # Fallback to CPU
            return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    except Exception as e:
        print(f"CUDA transform failed, using CPU fallback: {e}")
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(theta_rad), 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def calculate_mse_gpu(img1, img2):
    """Calculate MSE using optimized NumPy (CuPy removed)"""
    try:
        # Use NumPy for fast MSE calculation
        diff = img1.astype(np.float32) - img2.astype(np.float32)
        return float(np.mean(diff * diff))
    except Exception as e:
        print(f"MSE calculation failed: {e}")
        return float('inf')

def estimate_ego_motion(img1, img2):
    """Optimized ego motion estimation - reduced iterations for real-time performance"""
    # Reduce candidates and iterations for better performance
    NUM_CANDIDATES_FAST = 32  # Reduced from 64
    MAX_ITER_FAST = 15        # Reduced from 30
    
    # Initialize candidate population
    candidates = np.random.uniform(-3, 3, size=(NUM_CANDIDATES_FAST, 3))  # dx, dy, theta
    
    best_mse = float('inf')
    best_params = None
    
    for iteration in range(MAX_ITER_FAST):
        mses = []
        
        for dx, dy, theta in candidates:
            try:
                transformed = apply_ego_motion_cuda(img2, dx, dy, theta)
                mse = calculate_mse_gpu(img1, transformed)
                mses.append(mse)
            except Exception as e:
                mses.append(np.inf)
        
        mses = np.array(mses)
        best_idx = np.argsort(mses)[:NUM_CANDIDATES_FAST // 2]
        best_candidates = candidates[best_idx]
        
        current_best_mse = mses[best_idx[0]]
        if current_best_mse < best_mse:
            best_mse = current_best_mse
            best_params = candidates[best_idx[0]].copy()
        
        # Print less frequently to reduce overhead
        if iteration % 10 == 0:
            print(f"[Ego Motion Iter {iteration}] Best MSE: {current_best_mse:.6f}")
        
        if current_best_mse < MSE_THRESHOLD:
            break
        
        # Genetic operations
        # Mutation
        mutated = best_candidates + np.random.normal(0, MUTATION_STD * 0.8, best_candidates.shape)
        
        # Crossover
        if len(best_candidates) >= 2:
            crossover = (best_candidates[::2] + best_candidates[1::2]) / 2
        else:
            crossover = best_candidates
        
        # Rebuild candidate pool
        candidates = np.vstack((best_candidates, mutated, crossover))[:NUM_CANDIDATES_FAST]
    
    if best_params is not None:
        print(f"Final Ego Motion: dx={best_params[0]:.2f}, dy={best_params[1]:.2f}, yaw={np.degrees(best_params[2]):.2f}°")
    
    return best_params if best_params is not None else np.array([0, 0, 0])

def correct_flow_for_ego_motion(flow, ego_motion_params):
    """Correct optical flow for estimated ego motion"""
    if ego_motion_params is None:
        return flow
    
    dx, dy, theta = ego_motion_params
    h, w = flow.shape[:2]
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    center_x, center_y = w // 2, h // 2
    
    # Calculate ego motion flow
    # Translation component
    ego_flow_x = np.full((h, w), -dx, dtype=np.float32)
    ego_flow_y = np.full((h, w), -dy, dtype=np.float32)
    
    # Rotation component
    if abs(theta) > 1e-6:
        rel_x = x_coords - center_x
        rel_y = y_coords - center_y
        
        # Rotational flow components
        ego_flow_x += theta * rel_y
        ego_flow_y += -theta * rel_x
    
    # Subtract ego motion from observed flow
    corrected_flow = flow.copy()
    corrected_flow[:, :, 0] -= ego_flow_x
    corrected_flow[:, :, 1] -= ego_flow_y
    
    return corrected_flow

# ---------- UTILITY FUNCTIONS ---------- #
def is_object_in_roi(bbox):
    """Check if object center is inside the ROI"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    return (ROI_LEFT <= center_x <= ROI_RIGHT and 
            ROI_TOP <= center_y <= ROI_BOTTOM)

def get_ttc_threshold(bbox):
    """Get appropriate TTC threshold based on object location"""
    if is_object_in_roi(bbox):
        return TTC_THRESHOLD_INSIDE_ROI
    else:
        return TTC_THRESHOLD_OUTSIDE_ROI

def draw_roi(image):
    """Draw ROI rectangle on image"""
    cv2.rectangle(image, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), 
                  (255, 255, 0), 2)
    cv2.putText(image, f"ROI ({TTC_THRESHOLD_INSIDE_ROI}s)", (ROI_LEFT, ROI_TOP - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(image, f"Outside ROI: {TTC_THRESHOLD_OUTSIDE_ROI}s", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"Inside ROI: {TTC_THRESHOLD_INSIDE_ROI}s", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def compute_flow(prev_frame, curr_frame):
    """Compute optical flow between frames"""
    prev_resized = cv2.resize(prev_frame, (FRAME_WIDTH, FRAME_HEIGHT))
    curr_resized = cv2.resize(curr_frame, (FRAME_WIDTH, FRAME_HEIGHT))
    io_adapter = IOAdapter(pwcnet, (FRAME_HEIGHT, FRAME_WIDTH))
    inputs = io_adapter.prepare_inputs([prev_resized, curr_resized])
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        predictions = pwcnet(inputs)
    flow = predictions['flows'][0, 0].permute(1, 2, 0).cpu().numpy()
    return flow

def draw_flow_only(flow):
    """Visualize optical flow"""
    flow_rgb = flow_utils.flow_to_rgb(flow)
    return cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)

def compute_divergence(flow):
    """Compute divergence of flow field"""
    u, v = flow[..., 0], flow[..., 1]
    du_dx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dv_dy = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    return du_dx + dv_dy

def compute_object_ttc_improved(flow, div_map, bbox, fps=30):
    """Enhanced TTC calculation with ego motion correction"""
    x1, y1, x2, y2 = map(int, bbox)
    
    h, w = div_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Extract ROI
    flow_roi = flow[y1:y2, x1:x2]
    div_roi = div_map[y1:y2, x1:x2]
    
    if flow_roi.size == 0:
        return None
    
    # Method 1: Divergence-based TTC
    median_div = np.median(div_roi)
    mean_div = np.mean(div_roi)
    div_to_use = median_div if abs(median_div) > abs(mean_div) * 0.5 else mean_div
    
    if abs(div_to_use) > 1e-5:
        ttc_div = 1.0 / (abs(div_to_use) * fps) if div_to_use > 0 else None
    else:
        ttc_div = None
    
    # Method 2: Enhanced flow magnitude based TTC
    flow_mag = np.sqrt(flow_roi[..., 0]**2 + flow_roi[..., 1]**2)
    mean_flow_mag = np.mean(flow_mag[flow_mag > np.percentile(flow_mag, 50)])
    
    if mean_flow_mag > 0.1:
        obj_height = y2 - y1
        obj_width = x2 - x1
        obj_size = max(obj_height, obj_width)
        
        estimated_distance = (OBJECT_SIZE * FOCAL_LENGTH) / obj_size
        velocity_px_per_frame = mean_flow_mag
        velocity_m_per_sec = (velocity_px_per_frame * estimated_distance * fps) / FOCAL_LENGTH
        
        if velocity_m_per_sec > 0.1:
            ttc_flow = estimated_distance / velocity_m_per_sec
        else:
            ttc_flow = None
    else:
        ttc_flow = None
    
    # Method 3: Looming-based TTC
    bbox_area = (x2 - x1) * (y2 - y1)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    sample_points = [
        (center_x, center_y),
        (center_x - 5, center_y),
        (center_x + 5, center_y),
        (center_x, center_y - 5),
        (center_x, center_y + 5)
    ]
    
    radial_flows = []
    for px, py in sample_points:
        if 0 <= py < h and 0 <= px < w:
            flow_at_point = flow[py, px]
            dx, dy = px - center_x, py - center_y
            if dx != 0 or dy != 0:
                norm = np.sqrt(dx*dx + dy*dy)
                if norm > 0:
                    radial_comp = (flow_at_point[0] * dx + flow_at_point[1] * dy) / norm
                    radial_flows.append(abs(radial_comp))
    
    if radial_flows:
        avg_radial_flow = np.mean(radial_flows)
        if avg_radial_flow > 0.1:
            ttc_radial = np.sqrt(bbox_area) / (avg_radial_flow * fps * 2)
        else:
            ttc_radial = None
    else:
        ttc_radial = None
    
    # Combine methods with weighting
    valid_ttcs = []
    weights = []
    
    if ttc_div is not None and 0.1 < ttc_div < 30:
        valid_ttcs.append(ttc_div)
        weights.append(3.0)
    
    if ttc_flow is not None and 0.1 < ttc_flow < 30:
        valid_ttcs.append(ttc_flow)
        weights.append(2.0)
    
    if ttc_radial is not None and 0.1 < ttc_radial < 30:
        valid_ttcs.append(ttc_radial)
        weights.append(1.0)
    
    if valid_ttcs:
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        ttc_final = np.average(valid_ttcs, weights=weights)
        return ttc_final
    else:
        return None

def annotate_ttc_on_flow_image(flow_image, flow, div_map, tracks, fps, processing_time, class_names):
    """Annotate flow image with TTC and object information"""
    draw_roi(flow_image)
    
    cv2.putText(flow_image, f"Processing: {processing_time:.1f}ms", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Updated FPS display with pink color
    current_fps = 1000/max(processing_time, 1)
    cv2.putText(flow_image, f"FPS: {current_fps:.1f}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)  # Pink color (255, 0, 255)
    
    collision_warnings = []
    
    for tid, data in tracks.items():
        bbox = data['bbox']
        class_name = data.get('class_name', 'unknown')
        ttc = compute_object_ttc_improved(flow, div_map, bbox, fps)
        
        if ttc is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            threshold = get_ttc_threshold(bbox)
            in_roi = is_object_in_roi(bbox)
            
            if ttc <= threshold:
                collision_warnings.append({
                    'id': tid,
                    'ttc': ttc,
                    'in_roi': in_roi,
                    'threshold': threshold,
                    'class': class_name
                })
                
                if in_roi:
                    color = (0, 0, 255)  # Red - Critical
                    thickness = 4
                else:
                    color = (0, 100, 255)  # Orange - Warning
                    thickness = 3
                
                cv2.rectangle(flow_image, (x1, y1), (x2, y2), color, thickness)
                
                label = f"COLLISION! {class_name} TTC: {ttc:.1f}s"
                location_text = "ROI" if in_roi else "OUT"
                full_label = f"{label} ({location_text})"
                
                label_size = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(flow_image, (cx - label_size[0]//2 - 5, cy - 20), 
                             (cx + label_size[0]//2 + 5, cy), (0, 0, 0), -1)
                cv2.putText(flow_image, full_label, (cx - label_size[0]//2, cy - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                
                cv2.putText(flow_image, f"ID:{tid}", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
    if collision_warnings:
        warning_text = f"COLLISION WARNINGS: {len(collision_warnings)}"
        cv2.putText(flow_image, warning_text, (10, flow_image.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        for i, warning in enumerate(collision_warnings[:3]):
            warn_text = f"ID{warning['id']}: {warning['class']} {warning['ttc']:.1f}s"
            cv2.putText(flow_image, warn_text, (10, flow_image.shape[0] - 30 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

# ---------- TRACKING FUNCTIONS ---------- #
def create_kalman_filter(cx, cy):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    kf.statePost = kf.statePre.copy()
    return kf

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Tracking variables
next_id = 0
tracks = {}

def associate_detections(detections):
    matched, unmatched_tracks, unmatched_detections = [], list(tracks.keys()), list(range(len(detections)))
    if detections and tracks:
        iou_matrix = np.zeros((len(tracks), len(detections)))
        track_ids = list(tracks.keys())
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = iou(tracks[tid]['bbox'], det['bbox'])
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
        det = detections[det_idx]
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        tracks[tid]['kf'].correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        tracks[tid]['bbox'] = det['bbox']
        tracks[tid]['class_name'] = det['class_name']
        tracks[tid]['lost'] = 0
        tracks[tid]['age'] += 1

def update_unmatched_tracks(unmatched_tracks, flow):
    h, w = flow.shape[:2]
    for tid in unmatched_tracks:
        pred = tracks[tid]['kf'].predict()
        px, py = int(pred[0]), int(pred[1])
        x1, x2 = max(0, px - 10), min(w, px + 10)
        y1, y2 = max(0, py - 10), min(h, py + 10)
        if x2 > x1 and y2 > y1:
            avg_flow = flow[y1:y2, x1:x2].mean(axis=(0, 1))
            px += int(avg_flow[0])
            py += int(avg_flow[1])
        tracks[tid]['kf'].correct(np.array([[np.float32(px)], [np.float32(py)]]))
        tracks[tid]['lost'] += 1

def create_new_tracks(unmatched_detections, detections):
    global next_id
    for det_idx in unmatched_detections:
        det = detections[det_idx]
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        kf = create_kalman_filter(cx, cy)
        tracks[next_id] = {
            'kf': kf, 
            'bbox': det['bbox'], 
            'class_name': det['class_name'],
            'lost': 0, 
            'age': 1
        }
        next_id += 1

def remove_lost_tracks():
    to_remove = [tid for tid, t in tracks.items() if t['lost'] > MAX_TRACK_LOST]
    for tid in to_remove:
        del tracks[tid]

# ---------- MAIN LOOP ---------- #
# ---------- MAIN LOOP ---------- #
def main():
    global REAL_TIME_PLAYBACK, EGO_MOTION_ENABLED
    
    # Open webcam instead of video file
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {WEBCAM_INDEX}")
        print("Available camera indices to try: 0, 1, 2...")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual webcam properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if unable to get FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam FPS: {fps}, Resolution: {width}x{height}")
    
    prev_frame = None
    frame_count = 0
    ego_motion_params = None
    
    target_frame_time = 1.0 / fps
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            break
        
        frame_count += 1
        
        if frame_count % SKIP_FRAMES != 0:
            continue
            
        # Print progress every 10 seconds for webcam
        if frame_count % 300 == 0:  # Print every 300 frames (10 seconds at 30fps)
            print(f"Processed {frame_count} frames from webcam")
        
        frame_start_time = time.time()
        
        detections = []
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Compute optical flow
        if prev_frame is not None:
            flow = compute_flow(prev_frame, frame)
            
            # Ego motion correction
            if EGO_MOTION_ENABLED and frame_count % 15 == 0:  # Update ego motion every 15 frames
                try:
                    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (FRAME_WIDTH, FRAME_HEIGHT)), cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    ego_motion_params = estimate_ego_motion(prev_gray, curr_gray)
                except Exception as e:
                    print(f"Ego motion estimation failed: {e}")
                    ego_motion_params = np.array([0, 0, 0])
            
            # Apply ego motion correction to flow
            if EGO_MOTION_ENABLED and ego_motion_params is not None:
                flow = correct_flow_for_ego_motion(flow, ego_motion_params)
            
            div_map = compute_divergence(flow)
        else:
            flow = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2), dtype=np.float32)
            div_map = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        
        flow_vis = draw_flow_only(flow)
        
        # Enhanced detection with relevant classes only
        results = yolo(frame_resized, verbose=False)[0]
        if results.boxes is not None:
            for box in results.boxes:
                conf = box.conf.cpu().item()
                cls = int(box.cls.cpu().item())
                
                # Filter for relevant classes only
                if conf >= CONF_THRESHOLD and cls in RELEVANT_CLASSES:
                    bbox = box.xyxy.cpu().numpy().astype(int).flatten()
                    class_name = RELEVANT_CLASSES[cls]
                    detections.append({
                        'bbox': bbox,
                        'class_name': class_name,
                        'confidence': conf
                    })
        
        # Tracking
        matched, unmatched_tracks, unmatched_detections = associate_detections(detections)
        update_matched_tracks(matched, detections)
        update_unmatched_tracks(unmatched_tracks, flow)
        create_new_tracks(unmatched_detections, detections)
        remove_lost_tracks()
        
        # Calculate processing time
        processing_time = (time.time() - frame_start_time) * 1000
        
        # Annotate original frame with bounding boxes and ROI
        draw_roi(frame_resized)
        
        # Add ego motion info if enabled
        if EGO_MOTION_ENABLED and ego_motion_params is not None:
            ego_text = f"Ego Motion: dx={ego_motion_params[0]:.1f}, dy={ego_motion_params[1]:.1f}, yaw={np.degrees(ego_motion_params[2]):.1f}°"
            cv2.putText(frame_resized, ego_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw tracked objects on original frame
        for tid, data in tracks.items():
            bbox = data['bbox']
            class_name = data.get('class_name', 'unknown')
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on location
            if is_object_in_roi(bbox):
                bbox_color = (0, 255, 255)  # Yellow for objects in ROI
            else:
                bbox_color = (0, 255, 0)    # Green for objects outside ROI
            
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Draw class name and ID
            label = f"ID:{tid} {class_name}"
            cv2.putText(frame_resized, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1)
        
        # Annotate flow image with TTC
        annotate_ttc_on_flow_image(flow_vis, flow, div_map, tracks, fps, processing_time, RELEVANT_CLASSES)
        
        # Add detection count info
        detection_info = f"Detections: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(frame_resized, detection_info, (10, frame_resized.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display detection class breakdown
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        y_pos = 130
        for class_name, count in class_counts.items():
            class_text = f"{class_name}: {count}"
            cv2.putText(frame_resized, class_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 15
        
        # Display both windows
        cv2.imshow("Enhanced Vehicle Tracking with YOLOv9t", frame_resized)
        cv2.imshow("Ego-Motion Corrected Flow with TTC Analysis", flow_vis)
        
        prev_frame = frame.copy()
        
        # Real-time playback control
        if REAL_TIME_PLAYBACK:
            current_time = time.time()
            elapsed_time = current_time - last_time
            sleep_time = target_frame_time - elapsed_time
            
            if sleep_time > 0:
                key = cv2.waitKey(int(sleep_time * 1000)) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            
            last_time = current_time
        else:
            key = cv2.waitKey(1) & 0xFF
        
        # Keyboard controls
        if key == 27:  # ESC key
            break
        elif key == ord('p'):  # Pause
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
        elif key == ord('r'):  # Toggle real-time mode
            REAL_TIME_PLAYBACK = not REAL_TIME_PLAYBACK
            print(f"Real-time playback: {REAL_TIME_PLAYBACK}")
        elif key == ord('e'):  # Toggle ego motion correction
            EGO_MOTION_ENABLED = not EGO_MOTION_ENABLED
            print(f"Ego motion correction: {EGO_MOTION_ENABLED}")
        elif key == ord('s'):  # Save current frame
            timestamp = int(time.time())
            cv2.imwrite(f"frame_{timestamp}_original.jpg", frame_resized)
            cv2.imwrite(f"frame_{timestamp}_flow.jpg", flow_vis)
            print(f"Saved frames at timestamp {timestamp}")
        elif key == ord('h'):  # Show help
            print("\n=== KEYBOARD CONTROLS ===")
            print("ESC: Exit")
            print("P: Pause/Resume")
            print("R: Toggle real-time playback")
            print("E: Toggle ego motion correction")
            print("S: Save current frames")
            print("H: Show this help")
            print("========================\n")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n=== WEBCAM SESSION COMPLETE ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Final track count: {len(tracks)}")
    print("===============================")

if __name__ == "__main__":
    print("=== Enhanced Collision Detection System (Webcam) ===")
    print("Features:")
    print("- YOLOv9t object detection")
    print("- Focused on collision-relevant objects")
    print("- Ego motion correction using genetic algorithm")
    print("- Enhanced TTC calculation")
    print("- Real-time webcam processing with GPU acceleration")
    print(f"- Using webcam index: {WEBCAM_INDEX}")
    print("\nPress 'H' during execution for keyboard controls")
    print("================================================\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()