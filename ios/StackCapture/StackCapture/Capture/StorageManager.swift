import Foundation
import UIKit

// MARK: - Storage Manager

actor StorageManager {
    private let sessionURL: URL
    private let rgbDirectory: URL

    private var poses: [PoseFrame] = []
    private var rgbCount: UInt64 = 0

    init(sessionURL: URL) throws {
        self.sessionURL = sessionURL
        self.rgbDirectory = sessionURL.appendingPathComponent("rgb")

        // Create directories (no depth directory)
        let fm = FileManager.default
        try fm.createDirectory(at: sessionURL, withIntermediateDirectories: true)
        try fm.createDirectory(at: rgbDirectory, withIntermediateDirectories: true)
    }

    // MARK: - Frame Storage

    func storeFrame(_ data: FrameData) async throws {
        // Store RGB (already resized + JPEG encoded)
        let rgbFilename = String(format: "%06llu.jpg", data.rgbIndex)
        let rgbURL = rgbDirectory.appendingPathComponent(rgbFilename)
        try data.jpegData.write(to: rgbURL)
        rgbCount = data.rgbIndex + 1

        // Add pose (only if transform is available — ARKit mode)
        if let transform = data.transform {
            let pose = PoseFrame(
                timestamp: data.timestamp,
                rgbIndex: data.rgbIndex,
                depth: data.depth,
                transform: transform
            )
            poses.append(pose)
        }
    }

    // MARK: - Finalization

    func finalize(
        startTime: Date,
        rgbResolution: [Int],
        deviceModel: String,
        iosVersion: String,
        encoderReadings: [EncoderReading],
        bleConnected: Bool,
        hasVideo: Bool,
        captureSource: CaptureSource,
        imuReadings: [IMUReading] = [],
        cameraIntrinsics: CameraIntrinsics? = nil
    ) async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        // Sort poses by rgbIndex (async processing may have written them out of order)
        let sortedPoses = poses.sorted { $0.rgbIndex < $1.rgbIndex }

        // Write poses.json (empty array for raw capture — filled by SLAM later)
        let posesURL = sessionURL.appendingPathComponent("poses.json")
        let posesData = try encoder.encode(sortedPoses)
        try posesData.write(to: posesURL)

        // Write encoders.json (if any readings)
        if !encoderReadings.isEmpty {
            let encodersURL = sessionURL.appendingPathComponent("encoders.json")
            let encoderData = try encoder.encode(encoderReadings)
            try encoderData.write(to: encodersURL)
        }

        // Write imu.json (if any readings — ultrawide/raw mode)
        if !imuReadings.isEmpty {
            let imuURL = sessionURL.appendingPathComponent("imu.json")
            let imuData = try encoder.encode(imuReadings)
            try imuData.write(to: imuURL)
        }

        // Write calib.txt (camera intrinsics for SLAM)
        if let intrinsics = cameraIntrinsics {
            let calibURL = sessionURL.appendingPathComponent("calib.txt")
            try intrinsics.calibString.write(to: calibURL, atomically: true, encoding: .utf8)
        }

        // Calculate duration
        let duration: Double? = sortedPoses.last.map { $0.timestamp - (sortedPoses.first?.timestamp ?? 0) }

        // Write metadata.json
        let metadata = SessionMetadata(
            deviceModel: deviceModel,
            iosVersion: iosVersion,
            startTime: startTime,
            endTime: Date(),
            rgbResolution: rgbResolution,
            rgbFrameCount: rgbCount,
            poseCount: UInt64(poses.count),
            durationSeconds: duration,
            encoderCount: encoderReadings.isEmpty ? nil : encoderReadings.count,
            bleConnected: bleConnected,
            hasVideo: hasVideo,
            captureSource: captureSource,
            slamProcessed: captureSource == .iphoneArkit ? nil : false,
            cameraIntrinsics: cameraIntrinsics,
            imuCount: imuReadings.isEmpty ? nil : imuReadings.count
        )

        let metadataURL = sessionURL.appendingPathComponent("metadata.json")
        let metadataEncoder = JSONEncoder()
        metadataEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        metadataEncoder.dateEncodingStrategy = .iso8601
        let metadataData = try metadataEncoder.encode(metadata)
        try metadataData.write(to: metadataURL)
    }

    var frameCount: Int {
        poses.count
    }
}

// MARK: - Errors

enum StorageError: Error {
    case writeError
}
