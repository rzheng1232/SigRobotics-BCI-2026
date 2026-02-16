import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "Uno R4 Test"
CHAR_UUID = "abcdefab-1234-5678-1234-abcdefabcdef"

def on_notify(sender, data):
    print("Received:", data.decode("utf-8", errors="replace"))

async def main():
    print("Scanning for device...")
    devices = await BleakScanner.discover(timeout=5.0)

    target = None
    for d in devices:
        if d.name == DEVICE_NAME:
            target = d
            break

    if not target:
        print("Device not found")
        return

    print(f"Found {target.name} @ {target.address}")
    print("Connecting...")

    async with BleakClient(target.address) as client:
        print("Connected")

        await client.start_notify(CHAR_UUID, on_notify)

        while True:
            await asyncio.sleep(1)

asyncio.run(main())
