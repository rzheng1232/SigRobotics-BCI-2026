import cv2
from ultralytics import YOLO
import math
import time
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- One Euro Filter ---
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
        if t_e <= 0: return x
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
def calculate_elbow_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
def get_elbow_data(frame, model, conf_threshold=0.2):
    """Extracts raw elbow angle and draws annotations."""
    results = model(frame, verbose=False)
    if not results[0].keypoints or len(results[0].keypoints.data) == 0:
        return None

    kpts = results[0].keypoints.data[0].cpu().numpy()
    # Indices: 5=L Shoulder, 7=L Elbow, 9=L Wrist
    if all(kpts[i][2] > conf_threshold for i in [5, 7, 9]):
        angle = calculate_elbow_angle(kpts[5][:2], kpts[7][:2], kpts[9][:2])
        
        # Annotate Frame
        for idx in [5, 7, 9]:
            x, y = int(kpts[idx][0]), int(kpts[idx][1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{int(angle)}deg", (int(kpts[7][0])+15, int(kpts[7][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return angle
    return None
# --- YOLOv8 Settings ---
KEYPOINT_NAMES = {5: "L Shoulder", 7: "L Elbow", 9: "L Wrist"}
POSE_CONNECTIONS = [(5, 7), (7, 9)]

model = YOLO("yolov8n-pose.pt")

def process_and_draw(frame, kpts, conf_threshold=0.2):
    """Draws pose and returns the elbow angle if detected."""
    angle = None
    # Draw connections
    # for start, end in POSE_CONNECTIONS:
    #     if kpts[start][2] > conf_threshold and kpts[end][2] > conf_threshold:
    #         cv2.line(frame, (int(kpts[start][0]), int(kpts[start][1])), 
    #                  (int(kpts[end][0]), int(kpts[end][1])), (0, 255, 0), 2)
    
    # Calculate Angle if Shoulder(5), Elbow(7), and Wrist(9) are visible
    if all(kpts[i][2] > conf_threshold for i in [5, 7, 9]):
        shoulder = kpts[5][:2]
        elbow = kpts[7][:2]
        wrist = kpts[9][:2]
        angle = calculate_elbow_angle(shoulder, elbow, wrist)
        
        # Display Angle near Elbow
        cv2.putText(frame, f"Angle: {int(angle)}deg", 
                    (int(elbow[0]) + 15, int(elbow[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw keypoint circles
    for idx in [5, 7, 9]: # Only drawing arm joints for clarity
        if kpts[idx][2] > conf_threshold:
            x, y = int(kpts[idx][0]), int(kpts[idx][1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, KEYPOINT_NAMES[idx], (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return angle
def process_video(video_path, model):
    """Processes a single video and returns kinematic data."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    angles_smoothed, velocities_smoothed, frame_times = [], [], []
    filter_inst = None
    frame_count = 0

    print(f"Processing: {os.path.basename(video_path)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        t_current = frame_count #/ fps
        raw_angle = get_elbow_data(frame, model)

        if raw_angle is not None:
            if filter_inst is None:
                filter_inst = OneEuroFilter(t_current, raw_angle)
                smoothed_angle = raw_angle
            else:
                smoothed_angle = filter_inst(t_current, raw_angle)

            # Calculate Velocity
            vel = (smoothed_angle - angles_smoothed[-1]) * fps if angles_smoothed else 0
            
            angles_smoothed.append(smoothed_angle)
            velocities_smoothed.append(vel)
            frame_times.append(t_current)
        # cv2.imshow(f"Tracking: {os.path.basename(video_path)}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frame_times), np.array(angles_smoothed), np.array(velocities_smoothed), frame_count
def plot_kinematics(t, angles, velocities, title="Elbow Kinematics"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(t, angles, color='red', label='Angle (deg)')
    ax1.set_ylabel("Degrees")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, velocities, color='royalblue', label='Velocity (deg/s)')
    ax2.set_ylabel("Deg/Sec")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Setup
    YOLO_MODEL = YOLO("yolov8n-pose.pt")
    RECORDINGS_DIR = "recordings"
    
    # Get latest file
    cam0_files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "camera0_*.mp4")))
    if not cam0_files:
        print("No videos found!")
    else:
        # Process
        time_data, angle_data, vel_data = process_video(cam0_files[10], YOLO_MODEL)
        
        # Plot
        if len(time_data) > 0:
            plot_kinematics(time_data, angle_data, vel_data, title=f"Trial: {os.path.basename(cam0_files[10])}")
