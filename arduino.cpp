#include <ArduinoBLE.h>

BLEService msgService("12345678-1234-1234-1234-1234567890ab");
BLEStringCharacteristic msgChar(
  "abcdefab-1234-5678-1234-abcdefabcdef",
  BLERead | BLENotify,
  20
);

unsigned long lastSend = 0;
int counter = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!BLE.begin()) {
    while (1);
  }

  BLE.setLocalName("Uno R4 Test");
  BLE.setAdvertisedService(msgService);

  msgService.addCharacteristic(msgChar);
  BLE.addService(msgService);

  msgChar.writeValue("0");

  BLE.advertise();
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    while (central.connected()) {
      if (millis() - lastSend >= 1000) {
        lastSend = millis();

        msgChar.writeValue(String(counter));
        counter++;
      }

      BLE.poll();
    }
  }
}
