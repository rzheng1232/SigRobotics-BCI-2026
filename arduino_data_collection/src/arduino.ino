#include <ArduinoBLE.h>

BLEService msgService("12345678-1234-1234-1234-1234567890ab");
BLECharacteristic msgChar(
  "abcdefab-1234-5678-1234-abcdefabcdef",
  BLERead | BLENotify,
  16
);

unsigned long lastSend = 0;
int counter = 0;
float emgData[4] = {0.013, 0.321, 0.6767, 0.676767};

void setup() {
  Serial.begin(9600);
  // while (!Serial);
  if (!BLE.begin()) {
    while (1);
  }

  BLE.setLocalName("SigRoboArd");
  BLE.setAdvertisedService(msgService);

  msgService.addCharacteristic(msgChar);
  BLE.addService(msgService);
  msgChar.writeValue((byte*)emgData, sizeof(emgData));

  BLE.advertise();
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    // send something every 1000 ms
    while (central.connected()) {
      if (millis() - lastSend >= 1000) {
        lastSend = millis();

        msgChar.writeValue((uint8_t*)emgData, sizeof(emgData));
        counter++;
      }

      BLE.poll();
    }
  }
}