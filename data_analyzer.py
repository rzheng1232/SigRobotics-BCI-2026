import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import butter, filtfilt
import yolocam
from ultralytics import YOLO
import glob

import re

import cv2
directory = './recordings/'
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]
files = os.listdir(directory)
emg_data_files = sorted([file for file in files if file.startswith("emg_data_")], key=natural_sort_key)
cam0_files = sorted([file for file in  files if file.startswith("camera0_")], key=natural_sort_key)
cam2_files = sorted([file for file in files if file.startswith("camera2_")], key=natural_sort_key)
emg_data = []
cols = 1

for i in range(0, len(emg_data_files)):
    dat = pd.read_csv(directory+f"emg_data_{i+1}.csv")
    emg_data.append(dat["Channel 8"].to_numpy())

# run a moving window filter on the data, comapre each value to median in its immediatw window and remvoes the row if detect spike
sampling_rate = 80
window_size = 50

def apply_lowpass(rectified_data, cutoff=5, fs=80, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    envelope = filtfilt(b, a, rectified_data)
    return envelope
def apply_bandpass(raw_data, lowcut=10, highcut=39, fs=80, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    cleaned_signal = filtfilt(b, a, raw_data)
    return cleaned_signal
processed_emg = []
for data in emg_data:
    for t in range(len(data)):
        avg = 0
        if t < window_size/2:
            avg = np.average(data[:window_size])
        elif len(data)-t < window_size/2:
            avg = np.average(data[window_size:-1])
        else:
            avg = np.average(data[int(t-window_size/2):int(t+window_size/2)])

        if (np.abs(data[t] - avg) > 10e4):
            data[t] = avg
    rectified = np.abs(apply_bandpass(data))
    smoothed_envelope = apply_lowpass(rectified, cutoff=5, fs=80) 
    processed_emg.append(smoothed_envelope)

YOLO_MODEL = YOLO("yolov8n-pose.pt")
RECORDINGS_DIR = "recordings"

# Get latest file
cam0_files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "camera0_*.mp4")), key=natural_sort_key)
time_data, angle_data, vel_data, frame_counts = [], [], [], []

if not cam0_files:
    print("No videos found!")
else:

    for cam0_file in cam0_files:
        res = yolocam.process_video(cam0_file, YOLO_MODEL)
        time_data.append(res[0])
        angle_data.append(res[1])
        vel_data.append(apply_lowpass(res[2], cutoff=3, fs=30))
        frame_counts.append(res[3])
print(time_data[0])

# cols = 3
# rows = math.ceil(len(processed_emg) / cols)
# fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharey=True)
# axes = axes.flatten()

# for i in range(len(processed_emg)):
#     axes[i].plot(processed_emg[i], color='crimson', linewidth=0.7)
#     axes[i].set_title(f"Trial {i+1}: Processed EMG")
#     axes[i].grid(True, alpha=0.2)
    
#     # Label only the edges for cleanliness
#     if i >= (len(processed_emg) - cols): axes[i].set_xlabel("Samples")
#     if i % cols == 0: axes[i].set_ylabel("Amplitude (uV)")

# # Hide empty subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')


# plt.tight_layout()
# plt.show()
cols = 4 # Reduced to 2 columns for better visibility of dual axes
rows = math.ceil(len(processed_emg) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
axes = axes.flatten()

VIDEO_FPS = 30
for i in range(len(processed_emg)):
    ax_emg = axes[i]
    ax_angle = ax_emg.twinx()
    
    ax_vel = ax_emg.twinx()
    ax_vel.spines["right"].set_position(("axes", 1.25))
    
    v_time = np.array(time_data[i])
    v_time_s = v_time / VIDEO_FPS
    video_duration_s = frame_counts[i] / VIDEO_FPS
    n_samples_emg = len(processed_emg[i])
    emg_fs = n_samples_emg / video_duration_s
    emg_time_s = np.linspace(0, video_duration_s, n_samples_emg)
    # print(emg_fs) # Uncomment if you want to keep checking FS
    
    ax_emg.plot(emg_time_s, processed_emg[i], color='crimson', linewidth=0.8, alpha=0.7, label='EMG Envelope')
    ax_emg.set_ylabel("EMG Amplitude (uV)", color='crimson')
    ax_emg.tick_params(axis='y', labelcolor='crimson')
    
    angle = np.array(angle_data[i])
    vel = np.array(vel_data[i])
    
    synced_angle = np.interp(emg_time_s, v_time_s, angle)
    synced_vel = np.interp(emg_time_s, v_time_s, vel)
    
    ax_angle.plot(emg_time_s, synced_angle, color='royalblue', linewidth=2.0, label='Elbow Angle')
    ax_angle.set_ylabel("Angle (Degrees)", color='royalblue')
    ax_angle.tick_params(axis='y', labelcolor='royalblue')
    ax_angle.set_ylim(0, 180) 
    
    ax_vel.plot(emg_time_s, synced_vel, color='forestgreen', linewidth=1.5, alpha=0.8, label='Angular Velocity')
    ax_vel.set_ylabel("Velocity (Deg/s)", color='forestgreen')
    ax_vel.tick_params(axis='y', labelcolor='forestgreen')
    
    ax_emg.set_title(f"Trial {i+1}: Muscle vs. Movement")
    ax_emg.grid(True, alpha=0.2)
    ax_emg.set_xlabel("Time (seconds)")

# Cleanup: Hide unused subplots
for j in range(len(processed_emg), len(axes)):
    axes[j].axis('off')

# Use subplots_adjust to add horizontal space for the extra Y-axes before tight_layout
plt.subplots_adjust(wspace=0.6)
plt.tight_layout()
plt.show()