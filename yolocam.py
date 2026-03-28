import cv2
from ultralytics import YOLO
import math
import time
import os
import glob

# ─── One Euro Filter ───
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

class LandmarkSmoother:
    def __init__(self, indices, min_cutoff=0.7, beta=0.007):
        self.indices = indices
        self.filters = {}
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.initialized = False

    def smooth(self, keypoints, t):
        """keypoints: list of (x, y, conf) per keypoint index"""
        if not self.initialized:
            for idx in self.indices:
                x, y = keypoints[idx][:2]
                self.filters[(idx, 'x')] = OneEuroFilter(t, x, self.min_cutoff, self.beta)
                self.filters[(idx, 'y')] = OneEuroFilter(t, y, self.min_cutoff, self.beta)
            self.initialized = True
            return {idx: keypoints[idx][:2] for idx in self.indices}

        result = {}
        for idx in self.indices:
            x, y = keypoints[idx][:2]
            sx = self.filters[(idx, 'x')](t, x)
            sy = self.filters[(idx, 'y')](t, y)
            result[idx] = (sx, sy)
        return result

# ─── YOLOv8 COCO Keypoint Indices ───
# All 17 keypoints
KEYPOINT_NAMES = {
    0: "Nose",
    1: "L Eye", 2: "R Eye", 3: "L Ear", 4: "R Ear",
    5: "L Shoulder", 6: "R Shoulder",
    7: "L Elbow", 8: "R Elbow",
    9: "L Wrist", 10: "R Wrist",
    11: "L Hip", 12: "R Hip",
    13: "L Knee", 14: "R Knee",
    15: "L Ankle", 16: "R Ankle"
}

# Standard COCO pose connections
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)  # Right leg
]

# ─── Load Model ───
# Options: yolov8n-pose (fastest), yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose (most accurate)
model = YOLO("yolov8m-pose.pt")  # medium model = good balance

def draw_pose(frame, raw_kpts, conf_threshold=0.2):
    """Draw full body pose with all 17 keypoints and connections"""
    points = {}
    
    # Draw lines first (connections)
    for start, end in POSE_CONNECTIONS:
        if start < len(raw_kpts) and end < len(raw_kpts):
            start_conf = raw_kpts[start][2]
            end_conf = raw_kpts[end][2]
            if start_conf > conf_threshold and end_conf > conf_threshold:
                sx, sy = int(raw_kpts[start][0]), int(raw_kpts[start][1])
                ex, ey = int(raw_kpts[end][0]), int(raw_kpts[end][1])
                cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
    
    # Draw keypoint circles
    for idx in range(len(raw_kpts)):
        conf = raw_kpts[idx][2]
        if conf > conf_threshold:
            x, y = int(raw_kpts[idx][0]), int(raw_kpts[idx][1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            # Draw label
            name = KEYPOINT_NAMES.get(idx, f"KP{idx}")
            cv2.putText(frame, name, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# ─── Setup Videos ───
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

# smoother_0 = LandmarkSmoother(ARM_INDICES, min_cutoff=0.7, beta=0.007)
# smoother_2 = LandmarkSmoother(ARM_INDICES, min_cutoff=0.7, beta=0.007)

if not cap0.isOpened():
    print(f"ERROR: Could not open video {video_path_0}")
if not cap2.isOpened():
    print(f"ERROR: Could not open video {video_path_2}")

print("YOLOv8 Pose - Full Body Tracking")
print("Playing recorded videos from recordings directory")
print("Press 'q' to quit\n")

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
        
        results0 = model(frame0, verbose=False)
        if results0[0].keypoints is not None and len(results0[0].keypoints.data) > 0:
            kpts = results0[0].keypoints.data[0].cpu().numpy()
            avg_conf_0 = kpts[:, 2].mean()
            detected_0 = (kpts[:, 2] > 0.2).sum()
            stats0.append((avg_conf_0, detected_0, brightness0))
            
            print(f"[{frame_num}] CAM 0 ✓ Brightness: {brightness0:6.1f}, Contrast: {contrast0:5.1f}, Avg Conf: {avg_conf_0:.3f}, Keypoints: {int(detected_0)}/17", end="")
            draw_pose(frame0, kpts, conf_threshold=0.2)
        else:
            print(f"[{frame_num}] CAM 0 ✗ NO DETECTION (B:{brightness0:.1f}, C:{contrast0:.1f})", end="")
        
        cv2.imshow("Camera 0 - Full Pose", frame0)

    if ret2:
        # Analyze frame quality
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        brightness2 = gray2.mean()
        contrast2 = gray2.std()
        
        results2 = model(frame2, verbose=False)
        if results2[0].keypoints is not None and len(results2[0].keypoints.data) > 0:
            kpts = results2[0].keypoints.data[0].cpu().numpy()
            avg_conf_2 = kpts[:, 2].mean()
            detected_2 = (kpts[:, 2] > 0.2).sum()
            stats2.append((avg_conf_2, detected_2, brightness2))
            
            print(f"  |  CAM 2 ✓ Brightness: {brightness2:6.1f}, Contrast: {contrast2:5.1f}, Avg Conf: {avg_conf_2:.3f}, Keypoints: {int(detected_2)}/17")
            draw_pose(frame2, kpts, conf_threshold=0.2)
        else:
            print(f"  |  CAM 2 ✗ NO DETECTION (B:{brightness2:.1f}, C:{contrast2:.1f})")
        
        cv2.imshow("Camera 2 - Full Pose", frame2)

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
    print(f"Camera 0: Avg Conf: {avg_conf_0:.3f}, Avg Keypoints: {avg_kpts_0:.1f}/17, Avg Brightness: {avg_bright_0:.1f}")
else:
    print("Camera 0: NO DETECTIONS")

if stats2:
    avg_conf_2 = sum(s[0] for s in stats2) / len(stats2)
    avg_kpts_2 = sum(s[1] for s in stats2) / len(stats2)
    avg_bright_2 = sum(s[2] for s in stats2) / len(stats2)
    print(f"Camera 2: Avg Conf: {avg_conf_2:.3f}, Avg Keypoints: {avg_kpts_2:.1f}/17, Avg Brightness: {avg_bright_2:.1f}")
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