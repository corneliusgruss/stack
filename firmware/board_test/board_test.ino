/*
 * ESP32 Board Test - No external hardware needed
 * Tests: Serial output, LED blink, I2C bus scan
 *
 * If you see output on serial at 115200 baud, the board works.
 * I2C scan will show any devices connected to GPIO21 (SDA) / GPIO22 (SCL).
 */

#include <Wire.h>

#define LED_PIN 2  // Built-in LED on most ESP32 boards

void setup() {
  Serial.begin(115200);
  delay(1000);  // Wait for serial monitor

  pinMode(LED_PIN, OUTPUT);

  Serial.println("============================");
  Serial.println("  ESP32 Board Test");
  Serial.println("============================");
  Serial.println();

  // Print chip info
  Serial.printf("Chip model:    %s\n", ESP.getChipModel());
  Serial.printf("Chip cores:    %d\n", ESP.getChipCores());
  Serial.printf("CPU freq:      %d MHz\n", ESP.getCpuFreqMHz());
  Serial.printf("Flash size:    %d KB\n", ESP.getFlashChipSize() / 1024);
  Serial.printf("Free heap:     %d bytes\n", ESP.getFreeHeap());
  Serial.println();

  // Init I2C
  Wire.begin(21, 22);
  Wire.setClock(400000);

  // I2C scan
  Serial.println("I2C bus scan (SDA=21, SCL=22):");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  Found device at 0x%02X", addr);
      if (addr == 0x36) Serial.print(" (AS5600 encoder)");
      if (addr == 0x70) Serial.print(" (TCA9548A multiplexer)");
      Serial.println();
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  No I2C devices found (expected if nothing wired to GPIO21/22)");
  } else {
    Serial.printf("  %d device(s) found\n", found);
  }
  Serial.println();

  Serial.println("LED blinking on GPIO2 - check for blue light");
  Serial.println("Sending heartbeat every second...");
  Serial.println();
}

int count = 0;

void selectMuxChannel(uint8_t ch) {
  Wire.beginTransmission(0x70);
  Wire.write(1 << ch);
  Wire.endTransmission();
}

void i2cScan() {
  // Scan main bus
  Serial.println("--- Main bus ---");
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  0x%02X", addr);
      if (addr == 0x70) Serial.print(" = TCA9548A");
      Serial.println();
    }
  }

  // Scan behind each mux channel
  for (uint8_t ch = 0; ch < 8; ch++) {
    selectMuxChannel(ch);
    delayMicroseconds(100);
    for (uint8_t addr = 1; addr < 127; addr++) {
      if (addr == 0x70) continue;  // skip mux itself
      Wire.beginTransmission(addr);
      if (Wire.endTransmission() == 0) {
        Serial.printf("  CH%d: 0x%02X", ch, addr);
        if (addr == 0x36) Serial.print(" = AS5600 encoder!");
        Serial.println();
      }
    }
  }
  // Disable all channels
  Wire.beginTransmission(0x70);
  Wire.write(0);
  Wire.endTransmission();
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
  delay(500);

  count++;
  i2cScan();
}
