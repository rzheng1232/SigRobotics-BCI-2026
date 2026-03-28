import asyncio
import sys
import collections
from bleak import BleakScanner, BleakClient
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import qasync
import cv2
from ultralytics import YOLO
import math
import time

DEVICE_NAME = "SigRoboArd"
CHAR_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"
WINDOW_SIZE = 500
PACKET_SIZE = 135
BYTES_PER_SAMPLE = 27
NUM_CHANNELS = 8
LEFT_SHOULDER = 5
LEFT_ELBOW = 7
LEFT_WRIST = 9
ARM_INDICES = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]
ARM_CONNECTIONS = [(LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST)]
LABELS = {LEFT_SHOULDER: "Shoulder", LEFT_ELBOW: "Elbow", LEFT_WRIST: "Wrist"}
model = YOLO("yolov8m-pose.pt")  # medium model = good balance
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

# Create 8 separate queues, one for each channel
buffer = bytearray()
data_queues = [collections.deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(NUM_CHANNELS)]
raw_data = [[], [], [], [], [], [], [], []]
cleaner_data = [[], [], [], [], [], [], [], []]
start_collect = False
def decode_24bit_signed(b1, b2, b3):
    value = (b1 << 16) | (b2 << 8) | b3
    if value & 0x800000:  # sign bit
        value -= 1 << 24
    return value

def on_notify(sender, data):
    global buffer
    buffer.extend(data)
    
    while len(buffer) >= BYTES_PER_SAMPLE:
        # 1. FRAME SYNC
        if (buffer[0] & 0xF0) != 0xC0:
            buffer.pop(0) 
            continue 
            
        if len(buffer) < PACKET_SIZE:
            break

        # 2. Extract packet
        packet = buffer[:PACKET_SIZE]
        buffer = buffer[PACKET_SIZE:]
        
        for sample_start in range(0, PACKET_SIZE, BYTES_PER_SAMPLE):
            if (packet[sample_start] & 0xF0) != 0xC0:
                continue 
                
            base = sample_start + 3  # skip 3 status bytes
            
            # --- LOOP THROUGH ALL 8 CHANNELS ---
            for ch_index in range(NUM_CHANNELS):
                idx = base + ch_index * 3
                
                raw = decode_24bit_signed(packet[idx], packet[idx + 1], packet[idx + 2])
                microvolts = 1_000_000 * (4.5 / 8388607.0) * raw
                if (start_collect == True):
                    raw_data[ch_index].append(microvolts)
                # If we get a massive spike, just repeat the last known good value
                if (microvolts > 10000 or microvolts < -10000):
                    if len(data_queues[ch_index]) > 0:
                        microvolts = data_queues[ch_index][-1]
                    else:
                        microvolts = 0.0
                if (start_collect==True):
                    cleaner_data[ch_index].append(microvolts)
                data_queues[ch_index].append(microvolts)
                
        # Print all 8 channels neatly
        print(" | ".join(f"CH{i+1}: {data_queues[i][-1]:.0f} µV" for i in range(NUM_CHANNELS)))

async def ble_main():
    print("Scanning for device...")
    target = None
    i = 0
    while not target:
        devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
        for address, (device, adv_data) in devices.items():
            name = adv_data.local_name or device.name or "Unknown"
            if name == DEVICE_NAME:
                target = device
                break
        if not target:
            print(f"Device not found {i}")
            i += 1

    print(f"Found {target.name} @ {target.address}")
    print("Connecting...")
    async with BleakClient(target.address) as client:
        print("Connected")
        await client.start_notify(CHAR_UUID, on_notify)
        while True:
            await asyncio.sleep(0.05)
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
smoother_0 = LandmarkSmoother(ARM_INDICES, min_cutoff=0.7, beta=0.007)
smoother_2 = LandmarkSmoother(ARM_INDICES, min_cutoff=0.7, beta=0.007)
start_time = time.time()

async def video_main():
    t = time.time() - start_time
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()

    if ret0:
        results0 = model(frame0, verbose=False)
        if results0[0].keypoints is not None and len(results0[0].keypoints.data) > 0:
            kpts = results0[0].keypoints.data[0].cpu().numpy()
            smoothed = smoother_0.smooth(kpts, t)
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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    pg.setConfigOption('background', '#1a1a2e')
    pg.setConfigOption('foreground', '#eaeaea')

    win = pg.GraphicsLayoutWidget(title="BCI Signal Viewer — 8 Channels")
    win.resize(1400, 900) # Made the window big again
    win.show()

    COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    curves = []
    
    # Create 8 sub-plots dynamically
    for i in range(NUM_CHANNELS):
        p = win.addPlot(row=i, col=0, title=f"CH {i + 1}")
        p.setLabel('left', 'µV')
        p.setXRange(0, WINDOW_SIZE)
        p.setYRange(-5000, 5000)
        p.showGrid(x=True, y=True, alpha=0.2)
        # Only show the X-axis numbers on the very bottom plot to save space
        p.getAxis('bottom').setStyle(showValues=(i == NUM_CHANNELS - 1))
        
        curve = p.plot(pen=pg.mkPen(color=COLORS[i], width=1.5))
        curves.append(curve)

    def update():
        # Update all 8 curves
        for i, curve in enumerate(curves):
            raw_data = list(data_queues[i])
            
            if len(raw_data) > 0:
                dc_offset = sum(raw_data) / len(raw_data)
                centered_data = [x - dc_offset for x in raw_data]
                curve.setData(centered_data)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)  # 20 FPS

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    with loop:
        loop.create_task(ble_main())
        loop.create_task(video_main())
        loop.run_forever()
        