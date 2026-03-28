import asyncio
import sys
import collections
import csv
import time
from bleak import BleakScanner, BleakClient
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import qasync

DEVICE_NAME = "SigRoboArd"
CHAR_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"
WINDOW_SIZE = 500
PACKET_SIZE = 135
BYTES_PER_SAMPLE = 27
NUM_CHANNELS = 8

buffer = bytearray()
data_queues = [collections.deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(NUM_CHANNELS)]

# Global data collection variables
start_collect = False
raw_data_log = [[] for _ in range(NUM_CHANNELS)]
clean_data_log = [[] for _ in range(NUM_CHANNELS)]

def decode_24bit_signed(b1, b2, b3):
    value = (b1 << 16) | (b2 << 8) | b3
    if value & 0x800000:  # sign bit
        value -= 1 << 24
    return value

def on_notify(sender, data):
    global buffer
    global start_collect, raw_data_log, clean_data_log
    
    buffer.extend(data)
    
    while len(buffer) >= BYTES_PER_SAMPLE:
        if (buffer[0] & 0xF0) != 0xC0:
            buffer.pop(0) 
            continue 
            
        if len(buffer) < PACKET_SIZE:
            break

        packet = buffer[:PACKET_SIZE]
        buffer = buffer[PACKET_SIZE:]
        
        for sample_start in range(0, PACKET_SIZE, BYTES_PER_SAMPLE):
            if (packet[sample_start] & 0xF0) != 0xC0:
                continue 
                
            base = sample_start + 3 
            
            for ch_index in range(NUM_CHANNELS):
                idx = base + ch_index * 3
                
                raw = decode_24bit_signed(packet[idx], packet[idx + 1], packet[idx + 2])
                microvolts = 1_000_000 * (4.5 / 8388607.0) * raw
                
                if start_collect:
                    raw_data_log[ch_index].append(microvolts)
                
                # Spike filter
                if (microvolts > 10000 or microvolts < -10000):
                    if len(data_queues[ch_index]) > 0:
                        microvolts = data_queues[ch_index][-1]
                    else:
                        microvolts = 0.0
                        
                if start_collect:
                    clean_data_log[ch_index].append(microvolts)
                    
                data_queues[ch_index].append(microvolts)
                
        # Optional: Disable this print if it slows down data collection too much
        print(" | ".join(f"CH{i+1}: {data_queues[i][-1]:.0f} µV" for i in range(NUM_CHANNELS)))

def save_to_csv():
    global raw_data_log, clean_data_log
    
    # Generate a unique timestamp for the file names
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_filename = f"emg_raw_{timestamp}.csv"
    clean_filename = f"emg_clean_{timestamp}.csv"
    
    headers = [f"CH{i+1}" for i in range(NUM_CHANNELS)]

    # zip(*list) transposes our data so each channel is a vertical column instead of a horizontal row
    with open(raw_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(zip(*raw_data_log))

    with open(clean_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(zip(*clean_data_log))
        
    print(f"\n✅ Data Successfully Saved!")
    print(f"📄 {raw_filename} ({len(raw_data_log[0])} rows)")
    print(f"📄 {clean_filename} ({len(clean_data_log[0])} rows)\n")

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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    pg.setConfigOption('background', '#1a1a2e')
    pg.setConfigOption('foreground', '#eaeaea')

    win = pg.GraphicsLayoutWidget(title="BCI Signal Viewer — 8 Channels")
    win.resize(1400, 900) 
    
    # --- NEW: Keyboard Event Listener ---
    def keyPressEvent(event):
        global start_collect, raw_data_log, clean_data_log
        
        # Check if the Spacebar was pressed
        if event.key() == QtCore.Qt.Key.Key_Space: 
            print("ssss")
            if not start_collect:
                # Clear old data and START
                print("\n🔴 RECORDING STARTED... Press Space again to stop.")
                raw_data_log = [[] for _ in range(NUM_CHANNELS)]
                clean_data_log = [[] for _ in range(NUM_CHANNELS)]
                start_collect = True
            else:
                # STOP and SAVE
                print("\n⏹️ RECORDING STOPPED. Saving files...")
                start_collect = False
                save_to_csv()

    # Attach our custom key press listener to the window
    win.keyPressEvent = keyPressEvent
    # ------------------------------------

    win.show()

    COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    curves = []
    
    for i in range(NUM_CHANNELS):
        p = win.addPlot(row=i, col=0, title=f"CH {i + 1}")
        p.setLabel('left', 'µV')
        p.setXRange(0, WINDOW_SIZE)
        p.setYRange(-5000, 5000)
        p.showGrid(x=True, y=True, alpha=0.2)
        p.getAxis('bottom').setStyle(showValues=(i == NUM_CHANNELS - 1))
        
        curve = p.plot(pen=pg.mkPen(color=COLORS[i], width=1.5))
        curves.append(curve)

    def update():
        for i, curve in enumerate(curves):
            # Renamed to queue_data to avoid clashing with the global raw_data_log
            queue_data = list(data_queues[i]) 
            
            if len(queue_data) > 0:
                dc_offset = sum(queue_data) / len(queue_data)
                centered_data = [x - dc_offset for x in queue_data]
                curve.setData(centered_data)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50) 

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        loop.create_task(ble_main())
        loop.run_forever()