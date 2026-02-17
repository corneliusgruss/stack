"""
Encoder communication with ESP32 gripper controller.

Handles serial connection, calibration, and streaming of AS5600 encoder data.
"""

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import serial
import serial.tools.list_ports


@dataclass
class EncoderReading:
    """Single reading from all gripper encoders."""

    timestamp_ms: int  # ESP32 timestamp
    index_mcp: float   # Index finger MCP joint (degrees)
    index_pip: float   # Index finger PIP joint (degrees)
    three_finger_mcp: float  # 3-finger unit MCP joint (degrees)
    three_finger_pip: float  # 3-finger unit PIP joint (degrees)
    local_time: float  # Host timestamp (time.time())

    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "index_mcp": self.index_mcp,
            "index_pip": self.index_pip,
            "three_finger_mcp": self.three_finger_mcp,
            "three_finger_pip": self.three_finger_pip,
            "local_time": self.local_time,
        }

    def to_array(self) -> list[float]:
        """Return joint angles as array [index_mcp, index_pip, 3f_mcp, 3f_pip]."""
        return [self.index_mcp, self.index_pip, self.three_finger_mcp, self.three_finger_pip]


def find_esp32_port() -> Optional[str]:
    """Auto-detect ESP32 serial port."""
    ports = serial.tools.list_ports.comports()

    # Common ESP32 USB-serial chip identifiers
    esp32_ids = ["CP210", "CH340", "SLAB", "Silicon Labs", "USB Serial"]

    for port in ports:
        desc = f"{port.description} {port.manufacturer or ''}"
        if any(id in desc for id in esp32_ids):
            return port.device

    # Fallback to first available
    if ports:
        return ports[0].device

    return None


class EncoderReceiver:
    """Manages serial connection to ESP32 encoder reader."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port or find_esp32_port()
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self._log_file: Optional[Path] = None
        self._csv_writer = None
        self._file_handle = None

    def connect(self) -> bool:
        """Establish serial connection to ESP32."""
        if not self.port:
            raise ValueError("No serial port specified and auto-detect failed")

        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2)  # Wait for ESP32 reset
            self.serial.reset_input_buffer()
            return True
        except serial.SerialException as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Close serial connection and any open log files."""
        if self.serial:
            self.serial.close()
            self.serial = None
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def calibrate(self):
        """Set current encoder positions as zero reference."""
        if self.serial:
            self.serial.write(b"c")

    def start_logging(self, session_dir: Path):
        """Begin logging encoder data to CSV."""
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = session_dir / f"encoders_{timestamp}.csv"

        self._file_handle = open(self._log_file, "w", newline="")
        self._csv_writer = csv.writer(self._file_handle)
        self._csv_writer.writerow([
            "local_time", "esp_timestamp_ms",
            "index_mcp", "index_pip",
            "three_finger_mcp", "three_finger_pip"
        ])

    def stop_logging(self) -> Optional[Path]:
        """Stop logging and return path to log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        return self._log_file

    def read(self) -> Optional[EncoderReading]:
        """Read and parse one encoder sample."""
        if not self.serial:
            return None

        try:
            line = self.serial.readline().decode("utf-8").strip()
        except (serial.SerialException, UnicodeDecodeError):
            return None

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            return None

        # Parse: timestamp_ms,index_mcp,index_pip,three_mcp,three_pip
        parts = line.split(",")
        if len(parts) != 5:
            return None

        try:
            reading = EncoderReading(
                timestamp_ms=int(parts[0]),
                index_mcp=float(parts[1]),
                index_pip=float(parts[2]),
                three_finger_mcp=float(parts[3]),
                three_finger_pip=float(parts[4]),
                local_time=time.time(),
            )

            # Log if enabled
            if self._csv_writer:
                self._csv_writer.writerow([
                    reading.local_time,
                    reading.timestamp_ms,
                    reading.index_mcp,
                    reading.index_pip,
                    reading.three_finger_mcp,
                    reading.three_finger_pip,
                ])
                self._file_handle.flush()

            return reading

        except (ValueError, IndexError):
            return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
