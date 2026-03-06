import asyncio
import sys
import collections
from bleak import BleakScanner, BleakClient
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import qasync

DEVICE_NAME = "SigRoboArd"
CHAR_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"
WINDOW_SIZE = 500
PACKET_SIZE = 135
BYTES_PER_SAMPLE = 27

# We only need one queue now for Channel 8
buffer = bytearray()
data_queue = collections.deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)

def decode_24bit_signed(b1, b2, b3):
    value = (b1 << 16) | (b2 << 8) | b3
    if value & 0x800000:  # sign bit
        value -= 1 << 24
    return value

def on_notify(sender, data):
    global buffer;

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
                
            base = sample_start + 3 
            
            # --- ONLY LOOK AT CHANNEL 8 (Index 7) ---
            ch_index = 7 
            idx = base + ch_index * 3
            
            raw = decode_24bit_signed(packet[idx], packet[idx + 1], packet[idx + 2])
            microvolts = 1_000_000 * (4.5 / 8388607.0) * raw
            if (microvolts > 10000 or microvolts < -10000):
                microvolts = data_queue[-1]
            data_queue.append(microvolts)
                
        # Print only CH8 to keep the terminal clean
        print(f"CH8: {data_queue[-1]:.0f} µV")

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

    win = pg.GraphicsLayoutWidget(title="BCI Signal Viewer — CH8 Debug")
    win.resize(1000, 400) # Made the window smaller since it's only 1 plot
    win.show()

    # Create a single plot
    p = win.addPlot(row=0, col=0, title="CH 8 (Bicep/Tricep)")
    p.setLabel('left', 'µV')
    p.setXRange(0, WINDOW_SIZE)
    p.setYRange(-5000, 5000) # Change this if the signal is still too big/small
    p.showGrid(x=True, y=True, alpha=0.2)
    
    # Yellow line for visibility
    curve = p.plot(pen=pg.mkPen(color='#F7DC6F', width=2.0))

    def update():
        raw_data = list(data_queue)
        
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
        loop.run_forever()