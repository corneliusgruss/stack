import Foundation
import simd
import CoreVideo

// MARK: - Capture Source

enum CaptureSource: String, Codable {
    case iphoneArkit = "iphone_arkit"
    case iphoneUltrawide = "iphone_ultrawide"
    case stereoUsb = "stereo_usb"
}

// MARK: - Camera Intrinsics

struct CameraIntrinsics: Codable {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float

    /// Format as single-line calibration string: "fx fy cx cy"
    var calibString: String {
        "\(fx) \(fy) \(cx) \(cy)"
    }
}

// MARK: - IMU Reading

struct IMUReading: Codable {
    let timestamp: Double
    let accel: [Double]  // [x, y, z] in g's (CoreMotion userAcceleration, gravity removed)
    let gyro: [Double]   // [x, y, z] rad/s
}

// MARK: - Pose Frame

struct PoseFrame: Codable {
    let timestamp: Double
    let rgbIndex: UInt64
    let depth: Float?  // Single-point LiDAR depth (meters) — virtual ToF sensor
    let transform: [[Float]]  // 4x4 row-major

    init(timestamp: Double, rgbIndex: UInt64, depth: Float?, transform: simd_float4x4) {
        self.timestamp = timestamp
        self.rgbIndex = rgbIndex
        self.depth = depth
        self.transform = Self.toRowMajor(transform)
    }

    private static func toRowMajor(_ m: simd_float4x4) -> [[Float]] {
        // simd_float4x4 is column-major, convert to row-major for JSON
        return [
            [m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x],
            [m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y],
            [m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z],
            [m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w]
        ]
    }
}

// MARK: - Encoder Reading

struct EncoderReading: Codable {
    let timestamp: Double       // iPhone timestamp (Date().timeIntervalSince1970)
    let espTimestampMs: UInt32  // ESP32 millis()
    let indexMcp: Float
    let indexPip: Float
    let threeFingerMcp: Float
    let threeFingerPip: Float

    enum CodingKeys: String, CodingKey {
        case timestamp
        case espTimestampMs = "esp_timestamp_ms"
        case indexMcp = "index_mcp"
        case indexPip = "index_pip"
        case threeFingerMcp = "three_finger_mcp"
        case threeFingerPip = "three_finger_pip"
    }
}

// MARK: - Session Metadata

struct SessionMetadata: Codable {
    let deviceModel: String
    let iosVersion: String
    let startTime: Date
    let endTime: Date?
    let rgbResolution: [Int]  // [width, height]
    let rgbFrameCount: UInt64
    let poseCount: UInt64
    let durationSeconds: Double?
    let encoderCount: Int?
    let bleConnected: Bool?
    let hasVideo: Bool?
    let captureSource: CaptureSource?
    let slamProcessed: Bool?
    let cameraIntrinsics: CameraIntrinsics?
    let imuCount: Int?

    init(
        deviceModel: String = "",
        iosVersion: String = "",
        startTime: Date = Date(),
        endTime: Date? = nil,
        rgbResolution: [Int] = [480, 360],
        rgbFrameCount: UInt64 = 0,
        poseCount: UInt64 = 0,
        durationSeconds: Double? = nil,
        encoderCount: Int? = nil,
        bleConnected: Bool? = nil,
        hasVideo: Bool? = nil,
        captureSource: CaptureSource? = nil,
        slamProcessed: Bool? = nil,
        cameraIntrinsics: CameraIntrinsics? = nil,
        imuCount: Int? = nil
    ) {
        self.deviceModel = deviceModel
        self.iosVersion = iosVersion
        self.startTime = startTime
        self.endTime = endTime
        self.rgbResolution = rgbResolution
        self.rgbFrameCount = rgbFrameCount
        self.poseCount = poseCount
        self.durationSeconds = durationSeconds
        self.encoderCount = encoderCount
        self.bleConnected = bleConnected
        self.hasVideo = hasVideo
        self.captureSource = captureSource
        self.slamProcessed = slamProcessed
        self.cameraIntrinsics = cameraIntrinsics
        self.imuCount = imuCount
    }
}

// MARK: - Session Info (for listing)

struct SessionInfo: Identifiable {
    let id: String  // folder name
    let url: URL
    let metadata: SessionMetadata?
    let folderSize: Int64

    var displayName: String {
        // session_2026-01-31_153045 → Jan 31, 3:30 PM
        let parts = id.split(separator: "_")
        if parts.count >= 3,
           let date = Self.parseSessionDate(String(parts[1]), String(parts[2])) {
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            formatter.timeStyle = .short
            return formatter.string(from: date)
        }
        return id
    }

    var sizeString: String {
        ByteCountFormatter.string(fromByteCount: folderSize, countStyle: .file)
    }

    private static func parseSessionDate(_ dateStr: String, _ timeStr: String) -> Date? {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmmss"
        return formatter.date(from: "\(dateStr)_\(timeStr)")
    }
}

// MARK: - Frame Data (for async processing)

struct FrameData {
    let timestamp: Double
    let rgbIndex: UInt64
    let depth: Float?  // Single-point depth from LiDAR center
    let transform: simd_float4x4?  // nil for raw capture (poses from SLAM later)
    // Already-encoded data (copied immediately to avoid ARKit buffer recycling)
    let jpegData: Data
    // Full-resolution pixel buffer for video recording (retained)
    let pixelBuffer: CVPixelBuffer?
}
