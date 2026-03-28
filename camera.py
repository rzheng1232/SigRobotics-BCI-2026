import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import os
import urllib.request
import math
import time
import numpy as np

# ─── Model Setup ───
# Using "full" model for better accuracy + stability (swap to "heavy" if you want max accuracy)
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading full pose model...")
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
    # Extract coordinates as numpy arrays for vector math
    # smoothed_data[idx] = (x, y, z)
    shoulder = np.array(smoothed_data[LEFT_SHOULDER])
    elbow    = np.array(smoothed_data[LEFT_ELBOW])
    wrist    = np.array(smoothed_data[LEFT_WRIST])

    # Create vectors (Elbow is the vertex)
    ba = shoulder - elbow
    bc = wrist - elbow

    # Calculate the cosine of the angle using the dot product formula
    # cos(theta) = (a·b) / (|a|*|b|)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Clip to handle floating point errors outside [-1, 1]
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

# ─── Left Arm Landmarks ───
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
ARM_INDICES = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]
ARM_CONNECTIONS = [(LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST)]
LABELS = {LEFT_SHOULDER: "Shoulder", LEFT_ELBOW: "Elbow", LEFT_WRIST: "Wrist"}

def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

def draw_arm(frame, smoothed, raw_landmarks, h, w):
    points = {}
    for idx in ARM_INDICES:
        vis = raw_landmarks[idx].visibility if hasattr(raw_landmarks[idx], 'visibility') else 1.0
        if vis > 0.5:
            sx, sy, sz = smoothed[idx]
            cx, cy = int(sx * w), int(sy * h)
            points[idx] = (cx, cy)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    for start, end in ARM_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 3)

    for idx, name in LABELS.items():
        if idx in points:
            cv2.putText(frame, name, (points[idx][0]+10, points[idx][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            sx, sy, sz = smoothed[idx]
            print(f"  Left {name:8s} -> x: {sx:.3f}, y: {sy:.3f}, z: {sz:.3f}")

# ─── Setup ───
landmarker_0 = create_landmarker()
landmarker_2 = create_landmarker()

# Separate smoother per camera
# Tune: lower min_cutoff = smoother but more lag, higher beta = faster response
smoother_0 = LandmarkSmoother(ARM_INDICES, min_cutoff=0.7, beta=0.007)
smoother_2 = LandmarkSmoother(ARM_INDICES, min_cutoff=0.7, beta=0.007)

cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap0.isOpened():
    print("ERROR: Camera 0 not found")
if not cap2.isOpened():
    print("ERROR: Camera 2 not found")

print("Tracking LEFT arm (Shoulder -> Elbow -> Wrist)")
print("Using FULL model + One Euro Filter for stability")
print("Press 'q' to quit")

ts0 = 0
ts2 = 0
start_time = time.time()

while True:
    t = time.time() - start_time
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()

    if ret0:
        rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        mp_img0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb0)
        results0 = landmarker_0.detect_for_video(mp_img0, ts0)
        ts0 += 33

        if results0.pose_landmarks:
            h, w, _ = frame0.shape
            smoothed = smoother_0.smooth(results0.pose_landmarks[0], t)
            print("Camera 0:")
            draw_arm(frame0, smoothed, results0.pose_landmarks[0], h, w)

        cv2.imshow("Camera 0 - Left Arm", frame0)

    if ret2:
        rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        mp_img2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2)
        results2 = landmarker_2.detect_for_video(mp_img2, ts2)
        ts2 += 33

        if results2.pose_landmarks:
            h, w, _ = frame2.shape
            smoothed = smoother_2.smooth(results2.pose_landmarks[0], t)
            print(calculate_elbow_angle(smoothed))
            print("Camera 2:")
            draw_arm(frame2, smoothed, results2.pose_landmarks[0], h, w)

        cv2.imshow("Camera 2 - Left Arm", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
landmarker_0.close()
landmarker_2.close()