import asyncio
import struct, collections
from bleak import BleakScanner, BleakClient
import matplotlib.pyplot as plt
DEVICE_NAME = "SigRoboArd"
CHAR_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"
NUM_CHANNELS = 4
WINDOW_SIZE = 100
data_channels = [[0.0] * WINDOW_SIZE for _ in range(NUM_CHANNELS)]
data_queues = [collections.deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(NUM_CHANNELS)]
def on_notify(sender, data):
    print(len(data))
    decoded = struct.unpack('<ffff', data)
    for i, channel in enumerate(decoded):
        data_channels[i].append(channel)
        data_queues[i].append(channel)
    print(f"CH1: {decoded[0]:.2f} | CH2: {decoded[1]:.2f} | CH3: {decoded[2]:.2f} | CH4: {decoded[3]:.2f}")


async def main():
    print("Scanning for device...")
    target = None
    i = 0
    while not target:
        # Runs a periodic loop to look for device when device not found
        devices = await BleakScanner.discover(timeout=5.0, return_adv=True)

        for address, (device, adv_data) in devices.items():
            name = adv_data.local_name or device.name or "Unknown"
            if name != 'Unknown':
                print(name)
            if name == DEVICE_NAME:
                target = device
                break

        if not target:
            print(f"Device not found {i}")
            i+=1

    print(f"Found {target.name} @ {target.address}")
    print("Connecting...")
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots()
    lines = [ax.plot(list(data_queues[i]), label=f"CH {i+1}")[0] for i in range(NUM_CHANNELS)]

    ax.set_ylim(0, 1) # Adjust based on your sensor range
    ax.legend(loc='upper right')
    ax.set_title("Real-Time EMG (Matplotlib)")

    async with BleakClient(target.address) as client:
        print("Connected")

        await client.start_notify(CHAR_UUID, on_notify)

        while True:
            for i in range(NUM_CHANNELS):
                lines[i].set_ydata(list(data_queues[i]))
            fig.canvas.draw()
            fig.canvas.flush_events()
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass