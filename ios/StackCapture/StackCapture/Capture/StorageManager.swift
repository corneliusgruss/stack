import Foundation
import UIKit

// MARK: - Storage Manager

actor StorageManager {
    private let sessionURL: URL
    private let rgbDirectory: URL
    private let depthDirectory: URL

    private var poses: [PoseFrame] = []
    private var rgbCount: UInt64 = 0
    private var depthCount: UInt64 = 0

    init(sessionURL: URL) throws {
        self.sessionURL = sessionURL
        self.rgbDirectory = sessionURL.appendingPathComponent("rgb")
        self.depthDirectory = sessionURL.appendingPathComponent("depth")

        // Create directories
        let fm = FileManager.default
        try fm.createDirectory(at: sessionURL, withIntermediateDirectories: true)
        try fm.createDirectory(at: rgbDirectory, withIntermediateDirectories: true)
        try fm.createDirectory(at: depthDirectory, withIntermediateDirectories: true)
    }

    // MARK: - Frame Storage

    func storeFrame(_ data: FrameData) async throws {
        // Store RGB (already JPEG encoded)
        let rgbFilename = String(format: "%06llu.jpg", data.rgbIndex)
        let rgbURL = rgbDirectory.appendingPathComponent(rgbFilename)
        try data.jpegData.write(to: rgbURL)
        rgbCount = data.rgbIndex + 1

        // Store depth if available
        if let depthData = data.depthData, let depthIdx = data.depthIndex {
            let depthFilename = String(format: "%06llu.bin", depthIdx)
            let depthURL = depthDirectory.appendingPathComponent(depthFilename)
            try depthData.write(to: depthURL)
            depthCount = depthIdx + 1
        }

        // Add pose
        let pose = PoseFrame(
            timestamp: data.timestamp,
            rgbIndex: data.rgbIndex,
            depthIndex: data.depthIndex,
            transform: data.transform
        )
        poses.append(pose)
    }

    // MARK: - Finalization

    func finalize(startTime: Date, rgbResolution: [Int], depthResolution: [Int], deviceModel: String, iosVersion: String) async throws {
        // Sort poses by rgbIndex (async processing may have written them out of order)
        let sortedPoses = poses.sorted { $0.rgbIndex < $1.rgbIndex }

        // Write poses.json
        let posesURL = sessionURL.appendingPathComponent("poses.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let posesData = try encoder.encode(sortedPoses)
        try posesData.write(to: posesURL)

        // Calculate duration
        let duration: Double? = poses.last.map { $0.timestamp - (poses.first?.timestamp ?? 0) }

        // Write metadata.json
        let metadata = SessionMetadata(
            deviceModel: deviceModel,
            iosVersion: iosVersion,
            startTime: startTime,
            endTime: Date(),
            rgbResolution: rgbResolution,
            depthResolution: depthResolution,
            rgbFrameCount: rgbCount,
            depthFrameCount: depthCount,
            poseCount: UInt64(poses.count),
            durationSeconds: duration
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
