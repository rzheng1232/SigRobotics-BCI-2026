import asyncio
from bleak import BleakScanner, BleakClient

# DEVICE_NAME = "SigRoboArd"
# SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"


async def main():
    print("Searching for EEG Service...")
    
    # Use find_device_by_filter to look for the Service UUID
    target = await BleakScanner.find_device_by_filter(
        lambda d, ad: "12345678-1234-1234-1234-1234567890ab" in ad.service_uuids,
        timeout=10.0
    )

    if not target:
        print("Device not found. Is it already connected to another app?")
        return

    # Use a fallback if name is None
    device_name = target.name if target.name else "Unknown Name"
    print(f"Found Device: {device_name} @ {target.address}")
    
    async with BleakClient(target) as client:
        # Once connected, the client usually has the most up-to-date name
        print(f"Connected to: {device_name}")
        
        # You can also verify the connection state
        if client.is_connected:
            print("Successfully established link.")
if __name__ == "__main__":
    asyncio.run(main())