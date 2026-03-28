import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import os
import urllib.request
import math
import time
import numpy as np
import glob
from pathlib import Path

# ─── Model Setup ───
# Using "lite" model for FASTER processing (good balance of speed vs accuracy)
# Swap to "full" if you need higher accuracy and have time budget
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading lite pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done!")

# ─── One Euro Filter (reduces jitter while keeping responsiveness) ───
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.7, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x

        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat
def calculate_elbow_angle(smoothed_data):
    """Calculate angle between shoulder-elbow-wrist"""
    shoulder = np.array(smoothed_data[11])  # Left shoulder
    elbow    = np.array(smoothed_data[13])  # Left elbow
    wrist    = np.array(smoothed_data[15])  # Left wrist
    ba = shoulder - elbow
    bc = wrist - elbow
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

class LandmarkSmoother:
    """Manages One Euro Filters for multiple landmarks, each with x/y/z."""
    def __init__(self, landmark_indices, min_cutoff=0.7, beta=0.007):
        self.indices = landmark_indices
        self.filters = {}
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.initialized = False

    def smooth(self, landmarks, t):
        if not self.initialized:
            for idx in self.indices:
                lm = landmarks[idx]
                self.filters[(idx, 'x')] = OneEuroFilter(t, lm.x, self.min_cutoff, self.beta)
                self.filters[(idx, 'y')] = OneEuroFilter(t, lm.y, self.min_cutoff, self.beta)
                self.filters[(idx, 'z')] = OneEuroFilter(t, lm.z, self.min_cutoff, self.beta)
            self.initialized = True
            return {idx: (lm.x, lm.y, lm.z) for idx, lm in
                    ((i, landmarks[i]) for i in self.indices)}

        result = {}
        for idx in self.indices:
            lm = landmarks[idx]
            sx = self.filters[(idx, 'x')](t, lm.x)
            sy = self.filters[(idx, 'y')](t, lm.y)
            sz = self.filters[(idx, 'z')](t, lm.z)
            result[idx] = (sx, sy, sz)
        return result

# ─── MediaPipe Pose Landmarks (full body, 33 points) ───
# Standard MediaPipe BODY pose connections
MEDIAPIPE_LANDMARK_NAMES = {
    0: "Nose", 1: "L Eye Inner", 2: "L Eye", 3: "L Eye Outer", 4: "R Eye Inner", 5: "R Eye", 6: "R Eye Outer",
    7: "L Ear", 8: "R Ear",
    9: "Mouth Left", 10: "Mouth Right",
    11: "L Shoulder", 12: "R Shoulder",
    13: "L Elbow", 14: "R Elbow",
    15: "L Wrist", 16: "R Wrist",
    17: "L Pinky", 18: "R Pinky",
    19: "L Index", 20: "R Index",
    21: "L Thumb", 22: "R Thumb",
    23: "L Hip", 24: "R Hip",
    25: "L Knee", 26: "R Knee",
    27: "L Ankle", 28: "R Ankle",
    29: "L Heel", 30: "R Heel",
    31: "L Foot Index", 32: "R Foot Index"
}

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
    (11, 12),  # Shoulders
    (23, 24),  # Hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
    (11, 23), (12, 24),  # Torso
    (0, 1), (0, 2), (0, 4), (0, 5)  # Head
]

# All keypoint indices for full body
ALL_KEYPOINTS = list(range(33))

def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

def draw_pose(frame, smoothed, raw_landmarks, h, w):
    """Draw full body pose with all 33 keypoints and connections"""
    points = {}
    
    # Draw lines first (connections)
    for start, end in POSE_CONNECTIONS:
        if start < len(raw_landmarks) and end < len(raw_landmarks):
            vis_start = raw_landmarks[start].visibility if hasattr(raw_landmarks[start], 'visibility') else 1.0
            vis_end = raw_landmarks[end].visibility if hasattr(raw_landmarks[end], 'visibility') else 1.0
            
            if vis_start > 0.2 and vis_end > 0.2 and start in smoothed and end in smoothed:
                sx, sy, _ = smoothed[start]
                ex, ey, _ = smoothed[end]
                cv2.line(frame, (int(sx * w), int(sy * h)), (int(ex * w), int(ey * h)), (0, 255, 0), 2)
    
    # Draw keypoint circles
    for idx in ALL_KEYPOINTS:
        vis = raw_landmarks[idx].visibility if hasattr(raw_landmarks[idx], 'visibility') else 1.0
        
        if vis > 0.2 and idx in smoothed:
            sx, sy, sz = smoothed[idx]
            cx, cy = int(sx * w), int(sy * h)
            points[idx] = (cx, cy)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Draw label for key joints only (to avoid clutter)
            if idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                name = MEDIAPIPE_LANDMARK_NAMES.get(idx, f"KP{idx}")
                cv2.putText(frame, name, (cx + 5, cy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

# ─── Setup ───
landmarker_0 = create_landmarker()
landmarker_2 = create_landmarker()

# Separate smoother per camera (full body - all 33 landmarks)
smoother_0 = LandmarkSmoother(ALL_KEYPOINTS, min_cutoff=0.7, beta=0.007)
smoother_2 = LandmarkSmoother(ALL_KEYPOINTS, min_cutoff=0.7, beta=0.007)

# Load videos from recordings directory (correct path for Training folder)
recordings_dir = "arduino_data_collection/src/recordings"
if not os.path.exists(recordings_dir):
    print(f"ERROR: Recordings directory not found at {recordings_dir}")
    print(f"Current directory: {os.getcwd()}")
    exit(1)

# Find the most recent camera0 and camera2 video files
camera0_files = sorted(glob.glob(os.path.join(recordings_dir, "camera0_*.mp4")))
camera2_files = sorted(glob.glob(os.path.join(recordings_dir, "camera2_*.mp4")))

if not camera0_files or not camera2_files:
    print(f"ERROR: No video files found in {recordings_dir}")
    print(f"Found camera0 files: {camera0_files}")
    print(f"Found camera2 files: {camera2_files}")
    exit(1)

# Use the most recent files (last in sorted list)
video_path_0 = camera0_files[-1]
video_path_2 = camera2_files[-1]

print(f"Loading Camera 0 video: {video_path_0}")
print(f"Loading Camera 2 video: {video_path_2}")

cap0 = cv2.VideoCapture(video_path_0)
cap2 = cv2.VideoCapture(video_path_2)

if not cap0.isOpened():
    print(f"ERROR: Could not open video {video_path_0}")
if not cap2.isOpened():
    print(f"ERROR: Could not open video {video_path_2}")

print("MediaPipe Pose - Full Body Tracking (Lite Model - FASTER)")
print("Playing recorded videos from recordings directory")
print("Press 'q' to quit\n")

ts0 = 0
ts2 = 0
start_time = time.time()
frame_num = 0
stats0 = []
stats2 = []

while True:
    t = time.time() - start_time
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    frame_num += 1

    if ret0:
        # Analyze frame quality
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        brightness0 = gray0.mean()
        contrast0 = gray0.std()
        
        rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        mp_img0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb0)
        results0 = landmarker_0.detect_for_video(mp_img0, ts0)
        ts0 += 33
        
        if results0.pose_landmarks and len(results0.pose_landmarks) > 0:
            h, w, _ = frame0.shape
            landmarks = results0.pose_landmarks[0]
            smoothed = smoother_0.smooth(landmarks, t)
            
            # Count visible keypoints
            visible_count = sum(1 for lm in landmarks if (lm.visibility if hasattr(lm, 'visibility') else 1.0) > 0.2)
            avg_conf_0 = sum(lm.visibility if hasattr(lm, 'visibility') else 1.0 for lm in landmarks) / len(landmarks)
            stats0.append((avg_conf_0, visible_count, brightness0))
            
            print(f"[{frame_num}] CAM 0 ✓ Brightness: {brightness0:6.1f}, Contrast: {contrast0:5.1f}, Avg Conf: {avg_conf_0:.3f}, Keypoints: {visible_count}/33", end="")
            draw_pose(frame0, smoothed, landmarks, h, w)
        else:
            print(f"[{frame_num}] CAM 0 ✗ NO DETECTION (B:{brightness0:.1f}, C:{contrast0:.1f})", end="")
        
        cv2.imshow("Camera 0 - Full Pose (Lite)", frame0)

    if ret2:
        # Analyze frame quality
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        brightness2 = gray2.mean()
        contrast2 = gray2.std()
        
        rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        mp_img2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2)
        results2 = landmarker_2.detect_for_video(mp_img2, ts2)
        ts2 += 33
        
        if results2.pose_landmarks and len(results2.pose_landmarks) > 0:
            h, w, _ = frame2.shape
            landmarks = results2.pose_landmarks[0]
            smoothed = smoother_2.smooth(landmarks, t)
            
            # Count visible keypoints
            visible_count = sum(1 for lm in landmarks if (lm.visibility if hasattr(lm, 'visibility') else 1.0) > 0.2)
            avg_conf_2 = sum(lm.visibility if hasattr(lm, 'visibility') else 1.0 for lm in landmarks) / len(landmarks)
            stats2.append((avg_conf_2, visible_count, brightness2))
            
            print(f"  |  CAM 2 ✓ Brightness: {brightness2:6.1f}, Contrast: {contrast2:5.1f}, Avg Conf: {avg_conf_2:.3f}, Keypoints: {visible_count}/33")
            draw_pose(frame2, smoothed, landmarks, h, w)
        else:
            print(f"  |  CAM 2 ✗ NO DETECTION (B:{brightness2:.1f}, C:{contrast2:.1f})")
        
        cv2.imshow("Camera 2 - Full Pose (Lite)", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
if stats0:
    avg_conf_0 = sum(s[0] for s in stats0) / len(stats0)
    avg_kpts_0 = sum(s[1] for s in stats0) / len(stats0)
    avg_bright_0 = sum(s[2] for s in stats0) / len(stats0)
    print(f"Camera 0: Avg Conf: {avg_conf_0:.3f}, Avg Keypoints: {avg_kpts_0:.1f}/33, Avg Brightness: {avg_bright_0:.1f}")
else:
    print("Camera 0: NO DETECTIONS")

if stats2:
    avg_conf_2 = sum(s[0] for s in stats2) / len(stats2)
    avg_kpts_2 = sum(s[1] for s in stats2) / len(stats2)
    avg_bright_2 = sum(s[2] for s in stats2) / len(stats2)
    print(f"Camera 2: Avg Conf: {avg_conf_2:.3f}, Avg Keypoints: {avg_kpts_2:.1f}/33, Avg Brightness: {avg_bright_2:.1f}")
else:
    print("Camera 2: NO DETECTIONS")

if stats0 and stats2:
    conf_diff = abs(avg_conf_0 - avg_conf_2)
    brightness_diff = abs(avg_bright_0 - avg_bright_2)
    print(f"\nDifference: Conf delta: {conf_diff:.3f}, Brightness delta: {brightness_diff:.1f}")
    if brightness_diff > 20:
        print("⚠️  LARGE brightness difference - check camera focus/angles!")
print("="*80)

cap0.release()
cap2.release()
cv2.destroyAllWindows()
landmarker_0.close()
landmarker_2.close()