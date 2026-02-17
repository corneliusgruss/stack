# Gripper Firmware

## encoder_reader

ESP32 firmware to read 4x AS5600 magnetic encoders via TCA9548A I2C multiplexer.

### Hardware Setup

```
ESP32 Pin       -> Component
---------------------------------
GPIO21 (SDA)    -> TCA9548A SDA
GPIO22 (SCL)    -> TCA9548A SCL
3.3V            -> TCA9548A VCC, AS5600 VCC
GND             -> TCA9548A GND, AS5600 GND

TCA9548A Channel -> Encoder
---------------------------------
Channel 0        -> Index MCP
Channel 1        -> Index PIP
Channel 2        -> 3-Finger MCP
Channel 3        -> 3-Finger PIP
```

### Flashing

1. Install Arduino IDE or PlatformIO
2. Install ESP32 board support
3. Open `encoder_reader/encoder_reader.ino`
4. Select board: "ESP32 Dev Module"
5. Select port and upload

### Usage

Output is CSV over USB serial at 115200 baud:
```
timestamp_ms,index_mcp,index_pip,three_finger_mcp,three_finger_pip
1234,45.2,30.1,50.0,25.5
```

Send `c` to calibrate (set current position as zero).

### Testing Without All Encoders

If you only have some encoders connected, the others will read as error (-1.0).
The firmware still works - just ignore the missing channels.
