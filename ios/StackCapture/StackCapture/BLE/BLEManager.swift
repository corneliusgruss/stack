import Foundation
import CoreBluetooth
import Combine

/// Manages BLE connection to the StackGlove ESP32 encoder reader.
/// Auto-scans on init, auto-reconnects on disconnect.
class BLEManager: NSObject, ObservableObject {
    // MARK: - Published state

    @Published var connectionState: ConnectionState = .disconnected
    @Published var encoderValues: [Float] = [-1, -1, -1, -1]  // [indexMcp, indexPip, threeMcp, threePip]
    @Published var hasError: Bool = false  // Any encoder reading -1.0

    enum ConnectionState: String {
        case disconnected = "Disconnected"
        case scanning = "Scanning"
        case connecting = "Connecting"
        case connected = "Connected"
    }

    var isConnected: Bool { connectionState == .connected }

    // MARK: - BLE UUIDs (must match ESP32 firmware)

    private let serviceUUID = CBUUID(string: "4A980001-1234-5678-ABCD-57AC601A0E00")
    private let encoderCharUUID = CBUUID(string: "4A980002-1234-5678-ABCD-57AC601A0E00")
    private let commandCharUUID = CBUUID(string: "4A980003-1234-5678-ABCD-57AC601A0E00")

    // MARK: - CoreBluetooth

    private var centralManager: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var encoderCharacteristic: CBCharacteristic?
    private var commandCharacteristic: CBCharacteristic?

    // MARK: - Recording

    private var isRecordingEncoders = false
    private var recordedReadings: [EncoderReading] = []
    private let recordingLock = NSLock()

    // MARK: - Init

    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }

    // MARK: - Public API

    func calibrate() {
        guard let char = commandCharacteristic else { return }
        let data = Data([0x01])
        peripheral?.writeValue(data, for: char, type: .withResponse)
        print("BLE: Sent calibrate command")
    }

    func startRecording() {
        recordingLock.lock()
        recordedReadings = []
        isRecordingEncoders = true
        recordingLock.unlock()
    }

    func stopRecording() {
        recordingLock.lock()
        isRecordingEncoders = false
        recordingLock.unlock()
    }

    func getRecordedReadings() -> [EncoderReading] {
        recordingLock.lock()
        let readings = recordedReadings
        recordingLock.unlock()
        return readings
    }

    // MARK: - Private

    private func startScanning() {
        guard centralManager.state == .poweredOn else { return }
        connectionState = .scanning
        centralManager.scanForPeripherals(withServices: [serviceUUID], options: nil)
        print("BLE: Scanning for StackGlove...")
    }

    private func parseEncoderNotification(_ data: Data) {
        guard data.count >= 20 else { return }

        // [uint32 timestamp_ms, float x4]
        var espTimestamp: UInt32 = 0
        var angles: [Float] = [0, 0, 0, 0]

        (data as NSData).getBytes(&espTimestamp, range: NSRange(location: 0, length: 4))
        (data as NSData).getBytes(&angles[0], range: NSRange(location: 4, length: 4))
        (data as NSData).getBytes(&angles[1], range: NSRange(location: 8, length: 4))
        (data as NSData).getBytes(&angles[2], range: NSRange(location: 12, length: 4))
        (data as NSData).getBytes(&angles[3], range: NSRange(location: 16, length: 4))

        DispatchQueue.main.async {
            self.encoderValues = angles
            self.hasError = angles.contains(where: { $0 < 0 })
        }

        // Record if capturing
        recordingLock.lock()
        if isRecordingEncoders {
            let reading = EncoderReading(
                timestamp: Date().timeIntervalSince1970,
                espTimestampMs: espTimestamp,
                indexMcp: angles[0],
                indexPip: angles[1],
                threeFingerMcp: angles[2],
                threeFingerPip: angles[3]
            )
            recordedReadings.append(reading)
        }
        recordingLock.unlock()
    }
}

// MARK: - CBCentralManagerDelegate

extension BLEManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state == .poweredOn {
            startScanning()
        } else {
            print("BLE: Central manager state: \(central.state.rawValue)")
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                         advertisementData: [String: Any], rssi RSSI: NSNumber) {
        print("BLE: Found StackGlove (RSSI: \(RSSI))")
        self.peripheral = peripheral
        peripheral.delegate = self
        central.stopScan()

        connectionState = .connecting
        central.connect(peripheral, options: nil)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("BLE: Connected to StackGlove")
        peripheral.discoverServices([serviceUUID])
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        print("BLE: Disconnected from StackGlove")
        DispatchQueue.main.async {
            self.connectionState = .disconnected
            self.encoderValues = [-1, -1, -1, -1]
            self.hasError = true
        }
        self.encoderCharacteristic = nil
        self.commandCharacteristic = nil

        // Auto-reconnect after delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.startScanning()
        }
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        print("BLE: Failed to connect: \(error?.localizedDescription ?? "unknown")")
        DispatchQueue.main.async {
            self.connectionState = .disconnected
        }
        // Retry
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            self.startScanning()
        }
    }
}

// MARK: - CBPeripheralDelegate

extension BLEManager: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        for service in services {
            if service.uuid == serviceUUID {
                peripheral.discoverCharacteristics([encoderCharUUID, commandCharUUID], for: service)
            }
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics else { return }

        for char in characteristics {
            if char.uuid == encoderCharUUID {
                encoderCharacteristic = char
                peripheral.setNotifyValue(true, for: char)
                print("BLE: Subscribed to encoder notifications")
            } else if char.uuid == commandCharUUID {
                commandCharacteristic = char
                print("BLE: Found command characteristic")
            }
        }

        DispatchQueue.main.async {
            self.connectionState = .connected
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard characteristic.uuid == encoderCharUUID, let data = characteristic.value else { return }
        parseEncoderNotification(data)
    }
}
