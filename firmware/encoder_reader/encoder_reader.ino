/*
 * Gripper Encoder Reader - ESP32
 * Reads 4x AS5600 magnetic encoders via TCA9548A I2C multiplexer
 * Streams joint angles over Serial (USB) at 100Hz AND BLE at 50Hz
 *
 * Hardware:
 *   - ESP32 (HiLetgo or similar)
 *   - TCA9548A I2C multiplexer
 *   - 4x AS5600 magnetic encoders
 *
 * Wiring:
 *   ESP32 GPIO21 (SDA) -> TCA9548A SDA
 *   ESP32 GPIO22 (SCL) -> TCA9548A SCL
 *   TCA9548A Channel 0 -> Index MCP encoder
 *   TCA9548A Channel 1 -> Index PIP encoder
 *   TCA9548A Channel 2 -> 3-Finger MCP encoder
 *   TCA9548A Channel 3 -> 3-Finger PIP encoder
 *
 * BLE Service: "4A980001-1234-5678-ABCD-57AC601A0E00"
 *   Encoder characteristic (notify): 20 bytes [uint32 timestamp_ms, float x4 angles]
 *   Command characteristic (write): send 0x01 to calibrate
 *
 * Serial output format (CSV):
 *   timestamp_ms,index_mcp,index_pip,three_finger_mcp,three_finger_pip
 */

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// I2C addresses
#define TCA9548A_ADDR 0x70
#define AS5600_ADDR 0x36

// AS5600 registers
#define AS5600_RAW_ANGLE_H 0x0C
#define AS5600_RAW_ANGLE_L 0x0D

// Encoder channel mapping on multiplexer
#define CH_INDEX_MCP 0
#define CH_INDEX_PIP 1
#define CH_THREE_MCP 2
#define CH_THREE_PIP 3

// Timing
#define SAMPLE_RATE_HZ 100
#define SAMPLE_PERIOD_MS (1000 / SAMPLE_RATE_HZ)
#define BLE_NOTIFY_INTERVAL 2  // Send BLE every 2nd sample = 50 Hz

// BLE UUIDs
#define SERVICE_UUID        "4A980001-1234-5678-ABCD-57AC601A0E00"
#define ENCODER_CHAR_UUID   "4A980002-1234-5678-ABCD-57AC601A0E00"
#define COMMAND_CHAR_UUID   "4A980003-1234-5678-ABCD-57AC601A0E00"

// Calibration offsets (set these after calibration)
int16_t offset_index_mcp = 0;
int16_t offset_index_pip = 0;
int16_t offset_three_mcp = 0;
int16_t offset_three_pip = 0;

unsigned long last_sample_time = 0;
uint8_t sample_counter = 0;

// BLE state
BLEServer* pServer = NULL;
BLECharacteristic* pEncoderChar = NULL;
BLECharacteristic* pCommandChar = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// Forward declarations
void calibrate();

// BLE Server Callbacks
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("# BLE: Client connected");
  }

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("# BLE: Client disconnected");
  }
};

// Command Characteristic Callbacks
class CommandCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pCharacteristic) {
    String value = pCharacteristic->getValue();
    if (value.length() > 0 && value[0] == 0x01) {
      calibrate();
    }
  }
};

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);  // SDA=21, SCL=22 for ESP32
  Wire.setClock(400000);  // 400kHz I2C

  delay(100);

  // Check if multiplexer is present
  Wire.beginTransmission(TCA9548A_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: TCA9548A not found!");
    while(1) delay(1000);
  }

  // Initialize BLE
  BLEDevice::init("StackGlove");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  // Encoder data characteristic (notify)
  pEncoderChar = pService->createCharacteristic(
    ENCODER_CHAR_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pEncoderChar->addDescriptor(new BLE2902());

  // Command characteristic (write)
  pCommandChar = pService->createCharacteristic(
    COMMAND_CHAR_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  pCommandChar->setCallbacks(new CommandCallbacks());

  pService->start();

  // Start advertising
  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  // Helps with iPhone connections
  BLEDevice::startAdvertising();

  Serial.println("# Gripper Encoder Reader (BLE + Serial)");
  Serial.println("# Format: timestamp_ms,index_mcp,index_pip,three_finger_mcp,three_finger_pip");
  Serial.println("# Angles in degrees (0-360)");
  Serial.println("# Send 'c' to calibrate (set current position as zero)");
  Serial.println("# BLE advertising as 'StackGlove'");
  Serial.println("# Ready");
}

void selectChannel(uint8_t channel) {
  Wire.beginTransmission(TCA9548A_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

uint16_t readAS5600RawAngle() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(AS5600_RAW_ANGLE_H);
  Wire.endTransmission(false);

  Wire.requestFrom(AS5600_ADDR, 2);
  if (Wire.available() < 2) {
    return 0xFFFF;  // Error
  }

  uint8_t high = Wire.read();
  uint8_t low = Wire.read();
  return ((uint16_t)high << 8) | low;
}

float readEncoderDegrees(uint8_t channel, int16_t offset) {
  selectChannel(channel);
  delayMicroseconds(50);  // Small delay after channel switch

  uint16_t raw = readAS5600RawAngle();
  if (raw == 0xFFFF) {
    return -1.0;  // Error indicator
  }

  // AS5600 is 12-bit (0-4095)
  int16_t adjusted = (int16_t)raw - offset;
  if (adjusted < 0) adjusted += 4096;
  if (adjusted >= 4096) adjusted -= 4096;

  return adjusted * 360.0 / 4096.0;
}

void calibrate() {
  // Read current positions and set as zero
  selectChannel(CH_INDEX_MCP);
  delayMicroseconds(50);
  offset_index_mcp = readAS5600RawAngle();

  selectChannel(CH_INDEX_PIP);
  delayMicroseconds(50);
  offset_index_pip = readAS5600RawAngle();

  selectChannel(CH_THREE_MCP);
  delayMicroseconds(50);
  offset_three_mcp = readAS5600RawAngle();

  selectChannel(CH_THREE_PIP);
  delayMicroseconds(50);
  offset_three_pip = readAS5600RawAngle();

  Serial.println("# Calibrated - current position is now zero");
}

void loop() {
  // Check for serial calibration command
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'c' || cmd == 'C') {
      calibrate();
    }
  }

  // Handle BLE reconnection
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);  // Give BLE stack time to get ready
    BLEDevice::startAdvertising();
    Serial.println("# BLE: Restarted advertising");
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }

  unsigned long now = millis();
  if (now - last_sample_time >= SAMPLE_PERIOD_MS) {
    last_sample_time = now;

    float index_mcp = readEncoderDegrees(CH_INDEX_MCP, offset_index_mcp);
    float index_pip = readEncoderDegrees(CH_INDEX_PIP, offset_index_pip);
    float three_mcp = readEncoderDegrees(CH_THREE_MCP, offset_three_mcp);
    float three_pip = readEncoderDegrees(CH_THREE_PIP, offset_three_pip);

    // Always output to Serial (100 Hz)
    Serial.print(now);
    Serial.print(",");
    Serial.print(index_mcp, 2);
    Serial.print(",");
    Serial.print(index_pip, 2);
    Serial.print(",");
    Serial.print(three_mcp, 2);
    Serial.print(",");
    Serial.println(three_pip, 2);

    // Send BLE notification every 2nd sample (50 Hz)
    sample_counter++;
    if (deviceConnected && sample_counter >= BLE_NOTIFY_INTERVAL) {
      sample_counter = 0;

      // Pack: [uint32 timestamp_ms, float angle1, float angle2, float angle3, float angle4] = 20 bytes
      uint8_t buf[20];
      uint32_t ts = (uint32_t)now;
      memcpy(buf + 0, &ts, 4);
      memcpy(buf + 4, &index_mcp, 4);
      memcpy(buf + 8, &index_pip, 4);
      memcpy(buf + 12, &three_mcp, 4);
      memcpy(buf + 16, &three_pip, 4);

      pEncoderChar->setValue(buf, 20);
      pEncoderChar->notify();
    }
  }
}
