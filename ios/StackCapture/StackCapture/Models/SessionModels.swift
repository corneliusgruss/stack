import Foundation
import simd

// MARK: - Pose Frame

struct PoseFrame: Codable {
    let timestamp: Double
    let rgbIndex: UInt64
    let depthIndex: UInt64?
    let transform: [[Float]]  // 4x4 row-major

    init(timestamp: Double, rgbIndex: UInt64, depthIndex: UInt64?, transform: simd_float4x4) {
        self.timestamp = timestamp
        self.rgbIndex = rgbIndex
        self.depthIndex = depthIndex
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

// MARK: - Session Metadata

struct SessionMetadata: Codable {
    let deviceModel: String
    let iosVersion: String
    let startTime: Date
    let endTime: Date?
    let rgbResolution: [Int]  // [width, height]
    let depthResolution: [Int]  // [width, height]
    let rgbFrameCount: UInt64
    let depthFrameCount: UInt64
    let poseCount: UInt64
    let durationSeconds: Double?

    init(
        deviceModel: String = "",
        iosVersion: String = "",
        startTime: Date = Date(),
        endTime: Date? = nil,
        rgbResolution: [Int] = [1920, 1440],
        depthResolution: [Int] = [256, 192],
        rgbFrameCount: UInt64 = 0,
        depthFrameCount: UInt64 = 0,
        poseCount: UInt64 = 0,
        durationSeconds: Double? = nil
    ) {
        self.deviceModel = deviceModel
        self.iosVersion = iosVersion
        self.startTime = startTime
        self.endTime = endTime
        self.rgbResolution = rgbResolution
        self.depthResolution = depthResolution
        self.rgbFrameCount = rgbFrameCount
        self.depthFrameCount = depthFrameCount
        self.poseCount = poseCount
        self.durationSeconds = durationSeconds
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
    let depthIndex: UInt64?
    let transform: simd_float4x4
    // Already-encoded data (copied immediately to avoid ARKit buffer recycling)
    let jpegData: Data
    let depthData: Data?
}
