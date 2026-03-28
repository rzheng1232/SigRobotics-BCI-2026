import cv2
from ultralytics import YOLO
import math
import time

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
# 5=left_shoulder, 7=left_elbow, 9=left_wrist
LEFT_SHOULDER = 5
LEFT_ELBOW = 7
LEFT_WRIST = 9
ARM_INDICES = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]
ARM_CONNECTIONS = [(LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST)]
LABELS = {LEFT_SHOULDER: "Shoulder", LEFT_ELBOW: "Elbow", LEFT_WRIST: "Wrist"}

# ─── Load Model ───
# Options: yolov8n-pose (fastest), yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose (most accurate)
model = YOLO("yolov8m-pose.pt")  # medium model = good balance

def draw_arm(frame, smoothed, raw_kpts, conf_threshold=0.5):
    points = {}
    for idx in ARM_INDICES:
        conf = raw_kpts[idx][2]
        if conf > conf_threshold:
            sx, sy = smoothed[idx]
            cx, cy = int(sx), int(sy)
            points[idx] = (cx, cy)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    for start, end in ARM_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 3)

    for idx, name in LABELS.items():
        if idx in points:
            cv2.putText(frame, name, (points[idx][0]+10, points[idx][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print(f"  Left {name:8s} -> x: {points[idx][0]}, y: {points[idx][1]}")

# ─── Setup Cameras ───
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

smoother_0 = LandmarkSmoother(ARM_INDICES, min_cutoff=0.7, beta=0.007)
smoother_2 = LandmarkSmoother(ARM_INDICES, min_cutoff=0.7, beta=0.007)

if not cap0.isOpened():
    print("ERROR: Camera 0 not found")
if not cap2.isOpened():
    print("ERROR: Camera 2 not found")

print("YOLOv8 Pose - Tracking LEFT arm")
print("Press 'q' to quit")

start_time = time.time()

while True:
    t = time.time() - start_time
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()

    if ret0:
        results0 = model(frame0, verbose=False)
        if results0[0].keypoints is not None and len(results0[0].keypoints.data) > 0:
            # Get first person's keypoints: shape (17, 3) -> x, y, conf
            kpts = results0[0].keypoints.data[0].cpu().numpy()
            smoothed = smoother_0.smooth(kpts, t)
            print("Camera 0:")
            draw_arm(frame0, smoothed, kpts)
        cv2.imshow("Camera 0 - Left Arm (YOLOv8)", frame0)

    if ret2:
        results2 = model(frame2, verbose=False)
        if results2[0].keypoints is not None and len(results2[0].keypoints.data) > 0:
            kpts = results2[0].keypoints.data[0].cpu().numpy()
            smoothed = smoother_2.smooth(kpts, t)
            print("Camera 2:")
            draw_arm(frame2, smoothed, kpts)
        cv2.imshow("Camera 2 - Left Arm (YOLOv8)", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()