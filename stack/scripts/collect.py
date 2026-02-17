"""
Data collection script.

Synchronizes encoder data with iPhone ARKit stream and saves demonstrations.

Usage:
    stack-collect --session my_demo_01
    stack-collect --session my_demo_01 --no-video  # Encoders only
"""

import argparse
import sys
import time
from pathlib import Path

from stack.data.encoder import EncoderReceiver, find_esp32_port


def main():
    parser = argparse.ArgumentParser(description="Collect demonstration data")
    parser.add_argument("--session", required=True, help="Session name")
    parser.add_argument("--port", help="ESP32 serial port (auto-detect if not specified)")
    parser.add_argument("--no-video", action="store_true", help="Skip video recording")
    parser.add_argument("--data-dir", default="data/raw", help="Output directory")
    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    session_dir = data_dir / args.session
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Session: {args.session}")
    print(f"Output: {session_dir}")

    # Find ESP32
    port = args.port or find_esp32_port()
    if not port:
        print("ERROR: No ESP32 found. Connect device and try again.")
        sys.exit(1)

    print(f"ESP32: {port}")

    # Connect to encoder
    receiver = EncoderReceiver(port)
    if not receiver.connect():
        sys.exit(1)

    receiver.start_logging(session_dir)

    print("\n" + "=" * 50)
    print("RECORDING")
    print("=" * 50)
    print("Press 'c' + Enter to calibrate (set zero position)")
    print("Press 'q' + Enter to stop recording")
    print("=" * 50 + "\n")

    # TODO: Start iPhone recording via network command

    import threading

    running = True

    def input_handler():
        nonlocal running
        while running:
            try:
                cmd = input().strip().lower()
                if cmd == "c":
                    receiver.calibrate()
                    print("Calibrated!")
                elif cmd == "q":
                    running = False
            except EOFError:
                break

    input_thread = threading.Thread(target=input_handler, daemon=True)
    input_thread.start()

    # Main recording loop
    sample_count = 0
    start_time = time.time()

    try:
        while running:
            reading = receiver.read()
            if reading:
                sample_count += 1
                if sample_count % 100 == 0:  # Print every ~1 second
                    elapsed = time.time() - start_time
                    rate = sample_count / elapsed
                    print(
                        f"\rSamples: {sample_count} | "
                        f"Rate: {rate:.1f} Hz | "
                        f"Joints: [{reading.index_mcp:.1f}, {reading.index_pip:.1f}, "
                        f"{reading.three_finger_mcp:.1f}, {reading.three_finger_pip:.1f}]",
                        end=""
                    )

    except KeyboardInterrupt:
        pass

    # Cleanup
    receiver.disconnect()
    log_file = receiver.stop_logging()

    print(f"\n\nRecording complete!")
    print(f"Samples: {sample_count}")
    print(f"Duration: {time.time() - start_time:.1f}s")
    print(f"Saved to: {log_file}")


if __name__ == "__main__":
    main()
