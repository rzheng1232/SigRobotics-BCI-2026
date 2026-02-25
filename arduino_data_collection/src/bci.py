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
NUM_CHANNELS = 8

buffer = bytearray()
data_queues = [collections.deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(NUM_CHANNELS)]


def decode_24bit_signed(b1, b2, b3):
    value = (b1 << 16) | (b2 << 8) | b3
    if value & 0x800000:  # sign bit
        value -= 1 << 24
    return value


def on_notify(sender, data):
    global buffer
    buffer.extend(data)
    while len(buffer) >= PACKET_SIZE:
        packet = buffer[:PACKET_SIZE]
        buffer = buffer[PACKET_SIZE:]
        for sample_start in range(0, PACKET_SIZE, BYTES_PER_SAMPLE):
            base = sample_start + 3  # skip 3 status bytes
            for ch in range(NUM_CHANNELS):
                idx = base + ch * 3
                raw = decode_24bit_signed(packet[idx], packet[idx + 1], packet[idx + 2])
                microvolts = 1_000_000 * 4.5 * (raw / 16777215)
                data_queues[ch].append(microvolts)
        print(" | ".join(f"CH{i+1}: {data_queues[i][-1]:.2f} µV" for i in range(NUM_CHANNELS)))


async def ble_main():
    print("Scanning for device...")
    target = None
    i = 0
    while not target:
        devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
        for address, (device, adv_data) in devices.items():
            name = adv_data.local_name or device.name or "Unknown"
            print(name)
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

    win = pg.GraphicsLayoutWidget(title="BCI Signal Viewer — SigRobotics")
    win.resize(1400, 900)
    win.show()

    COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

    curves = []
    for i in range(NUM_CHANNELS):
        p = win.addPlot(row=i, col=0, title=f"CH {i + 1}")
        p.setLabel('left', 'µV')
        p.setXRange(0, WINDOW_SIZE)
        p.setYRange(-500_000, 500_000)
        p.showGrid(x=True, y=True, alpha=0.2)
        # Only show x-axis tick labels on the bottom plot
        p.getAxis('bottom').setStyle(showValues=(i == NUM_CHANNELS - 1))
        curve = p.plot(pen=pg.mkPen(color=COLORS[i], width=1.5))
        curves.append(curve)

    def update():
        for i, curve in enumerate(curves):
            curve.setData(list(data_queues[i]))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)  # 20 FPS

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        loop.create_task(ble_main())
        loop.run_forever()