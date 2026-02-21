#include <ArduinoBLE.h>
#include <SPI.h>

// ardEEG SPI pins
const int chip_select = 10;
const int button_pin = 7;
const int drdy_pin = 5;
const int size_of_data = 1350;
byte output[size_of_data] = {};
const int ble_chunk_size = 180;

BLEService eegService("12345678-1234-1234-1234-1234567890ab");
BLECharacteristic eegChar(
  "abcdefab-1234-5678-1234-abcdefabcdef",
  BLERead | BLENotify,
  ble_chunk_size
);

int button_state = 0;
int sc = 0;
bool capture_armed = false;

void sendCommand(byte command) {
  digitalWrite(chip_select, LOW);
  SPI.transfer(command);
  digitalWrite(chip_select, HIGH);
}

void writeByte(byte registers, byte data) {
  byte spi_data = 0x40 | registers;
  byte spi_data_array[3];
  spi_data_array[0] = spi_data;
  spi_data_array[1] = 0x00;
  spi_data_array[2] = data;
  digitalWrite(chip_select, LOW);
  SPI.transfer(spi_data_array, 3);
  digitalWrite(chip_select, HIGH);
}

byte readEegByte() {
  digitalWrite(chip_select, LOW);
  byte value = SPI.transfer(0xFF);
  digitalWrite(chip_select, HIGH);
  return value;
}

void sendOutputOverBle() {
  int index = 0;
  while (index < size_of_data) {
    int chunk = size_of_data - index;
    if (chunk > ble_chunk_size) {
      chunk = ble_chunk_size;
    }
    eegChar.writeValue(output + index, chunk);
    index += chunk;
    BLE.poll();
    delay(2);
  }
}

void setup() {
  pinMode(button_pin, INPUT);
  pinMode(drdy_pin, INPUT);
  pinMode(chip_select, OUTPUT);
  digitalWrite(chip_select, HIGH);

  SPI.begin();
  SPI.beginTransaction(SPISettings(600000, MSBFIRST, SPI_MODE1));
  sendCommand(0x02); // wakeup
  sendCommand(0x0A); // stop
  sendCommand(0x06); // reset
  delay(2);
  sendCommand(0x11); // sdatac

  // Write configurations
  writeByte(0x01, 0x96);
  writeByte(0x02, 0xD4);
  writeByte(0x03, 0xFF);
  writeByte(0x04, 0x00);
  writeByte(0x0D, 0x00);
  writeByte(0x0E, 0x00);
  writeByte(0x0F, 0x00);
  writeByte(0x10, 0x00);
  writeByte(0x11, 0x00);
  writeByte(0x15, 0x20);
  writeByte(0x17, 0x00);
  writeByte(0x05, 0x00);
  writeByte(0x06, 0x00);
  writeByte(0x07, 0x00);
  writeByte(0x08, 0x00);
  writeByte(0x09, 0x00);
  writeByte(0x0A, 0x00);
  writeByte(0x0B, 0x00);
  writeByte(0x0C, 0x00);
  writeByte(0x14, 0x80);
  sendCommand(0x10);
  sendCommand(0x08);

  // BLE setup
  Serial.begin(9600);
  if (!BLE.begin()) {
    while (1);
  }
  BLE.setLocalName("SigRoboArd");
  BLE.setAdvertisedService(eegService);
  eegService.addCharacteristic(eegChar);
  BLE.addService(eegService);
  eegChar.writeValue(output, sizeof(output));
  BLE.advertise();
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    while (central.connected()) {
      button_state = digitalRead(button_pin);

      if (button_state == HIGH) {
        capture_armed = true;
      }

      if (capture_armed && button_state == LOW && digitalRead(drdy_pin) == LOW) {
        capture_armed = false;
        for (int i = 0; i < 27; i++) {
          if (sc < size_of_data) {
            output[sc] = readEegByte();
            sc = sc + 1;
          }
        }

        if (sc == size_of_data) {
          sendOutputOverBle();
          sc = 0;
        }
      }

      BLE.poll();
    }

    sc = 0;
    capture_armed = false;
  }
}