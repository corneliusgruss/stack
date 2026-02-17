import Foundation
import ARKit
import Combine
import UIKit
import CoreImage

// MARK: - Capture Coordinator

class CaptureCoordinator: ObservableObject {
    // MARK: - Published (must update on main thread)
    @Published var isRecording = false
    @Published var frameCount: UInt64 = 0
    @Published var duration: TimeInterval = 0
    @Published var error: Error?
    @Published var currentDepthPoint: Float? = nil

    /// Called from ARKit's thread with each camera pixel buffer for live preview display.
    var onPreviewBuffer: ((CVPixelBuffer) -> Void)?

    // MARK: - Main thread state
    var bleManager: BLEManager?
    private var storageManager: StorageManager?
    private var videoRecorder: VideoRecorder?
    private var startTime: Date?
    private let rgbResolution: [Int] = [480, 360]

    // MARK: - Processing queue and its state
    private let processingQueue = DispatchQueue(label: "capture.processing", qos: .userInitiated)
    // Only accessed on processingQueue:
    private var firstTimestamp: Double?
    private var rgbIndex: UInt64 = 0
    private var frameBuffer: [FrameData] = []
    private var pendingStorageTask: Task<Void, Never>?

    // Written on main thread, read on ARKit/processing thread (benign race at boundaries)
    private var recordingFlag = false

    // MARK: - Constants
    private let jpegQuality: CGFloat = 0.8
    private let ciContext = CIContext()
    private let targetWidth: Int = 480
    private let targetHeight: Int = 360
    private let batchSize = 10

    // MARK: - Public API (main thread)

    @MainActor
    func startCapture() throws {
        let sessionName = Self.generateSessionName()
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sessionURL = documentsURL.appendingPathComponent(sessionName)

        storageManager = try StorageManager(sessionURL: sessionURL)

        let videoURL = sessionURL.appendingPathComponent("video.mov")
        videoRecorder = VideoRecorder(outputURL: videoURL)

        startTime = Date()
        frameCount = 0
        duration = 0
        error = nil

        // Reset processing queue state before enabling recording
        processingQueue.sync {
            firstTimestamp = nil
            rgbIndex = 0
            frameBuffer = []
            pendingStorageTask = nil
        }

        bleManager?.startRecording()

        recordingFlag = true
        isRecording = true
        print("Started capture: \(sessionName)")
    }

    @MainActor
    func stopCapture() async {
        recordingFlag = false
        isRecording = false

        bleManager?.stopRecording()

        // Wait for processing queue to drain and grab remaining frames + pending task
        let (remaining, pendingTask): ([FrameData], Task<Void, Never>?) = await withCheckedContinuation { continuation in
            processingQueue.async { [self] in
                let frames = frameBuffer
                let task = pendingStorageTask
                frameBuffer = []
                pendingStorageTask = nil
                continuation.resume(returning: (frames, task))
            }
        }

        // Wait for any in-flight storage batch
        await pendingTask?.value

        // Flush remaining frames to storage
        if let storage = storageManager {
            for frame in remaining {
                do {
                    try await storage.storeFrame(frame)
                } catch {
                    print("Failed to store frame \(frame.rgbIndex): \(error)")
                    self.error = error
                }
            }
        }

        // Finalize video
        if let recorder = videoRecorder {
            await recorder.finish()
        }

        // Get encoder readings
        let encoderReadings = bleManager?.getRecordedReadings() ?? []

        guard let storage = storageManager, let start = startTime else { return }

        do {
            try await storage.finalize(
                startTime: start,
                rgbResolution: rgbResolution,
                deviceModel: ARSessionManager.deviceModel,
                iosVersion: UIDevice.current.systemVersion,
                encoderReadings: encoderReadings,
                bleConnected: bleManager?.isConnected ?? false,
                hasVideo: videoRecorder != nil
            )
            print("Capture finalized: \(frameCount) RGB frames, \(encoderReadings.count) encoder readings")
        } catch {
            self.error = error
            print("Failed to finalize: \(error)")
        }

        storageManager = nil
        videoRecorder = nil
        startTime = nil
    }

    // MARK: - Frame Handling (called from ARKit's thread)

    func handleFrame(_ frame: ARFrame) {
        // --- Fast extraction on ARKit's thread (<1ms) ---
        let timestamp = frame.timestamp
        let transform = frame.camera.transform
        let pixelBuffer = frame.capturedImage  // independently ref-counted

        // Sample depth point (fast — few pointer reads)
        let depthPoint: Float?
        if let depthMap = frame.smoothedSceneDepth?.depthMap {
            depthPoint = sampleCenterDepth(depthMap)
        } else {
            depthPoint = nil
        }

        // Live depth display (UI update)
        DispatchQueue.main.async { [weak self] in
            self?.currentDepthPoint = depthPoint
        }

        // Live preview — caller (ARKit thread) passes pixel buffer to display layer
        onPreviewBuffer?(pixelBuffer)

        // After this point, ARFrame can be released — we only hold pixelBuffer ref
        guard recordingFlag else { return }

        // --- Dispatch heavy work to serial processing queue ---
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            self.processFrame(
                timestamp: timestamp,
                transform: transform,
                pixelBuffer: pixelBuffer,
                depthPoint: depthPoint
            )
        }
    }

    // MARK: - Processing (on processingQueue — serial, no locking needed)

    private func processFrame(
        timestamp: Double,
        transform: simd_float4x4,
        pixelBuffer: CVPixelBuffer,
        depthPoint: Float?
    ) {
        // Initialize video recorder on first frame
        if firstTimestamp == nil {
            firstTimestamp = timestamp
            let fullWidth = CVPixelBufferGetWidth(pixelBuffer)
            let fullHeight = CVPixelBufferGetHeight(pixelBuffer)
            do {
                try videoRecorder?.start(width: fullWidth, height: fullHeight)
            } catch {
                print("Failed to start video recorder: \(error)")
            }
        }

        let currentDuration = timestamp - (firstTimestamp ?? timestamp)
        let currentIndex = rgbIndex
        rgbIndex += 1

        // UI updates (lightweight — just property writes)
        DispatchQueue.main.async { [weak self] in
            self?.duration = currentDuration
            self?.frameCount = currentIndex + 1
        }

        // Append full-res frame to video recorder (hardware HEVC — fast)
        videoRecorder?.appendFrame(pixelBuffer, timestamp: timestamp)

        // JPEG encode at 480x360 for training data (~5ms — the heavy operation)
        guard let jpegData = resizeAndEncodeToJPEG(pixelBuffer) else {
            print("Failed to encode frame \(currentIndex) to JPEG")
            return
        }

        // pixelBuffer can now be released (JPEG data is a copy, video encoder has its own ref)

        let frameData = FrameData(
            timestamp: timestamp,
            rgbIndex: currentIndex,
            depth: depthPoint,
            transform: transform,
            jpegData: jpegData,
            pixelBuffer: nil
        )

        // Queue for async storage write
        frameBuffer.append(frameData)

        if frameBuffer.count >= batchSize && pendingStorageTask == nil {
            let framesToStore = frameBuffer
            frameBuffer = []

            pendingStorageTask = Task { [weak self] in
                guard let self = self, let storage = self.storageManager else { return }
                for frame in framesToStore {
                    do {
                        try await storage.storeFrame(frame)
                    } catch {
                        print("Failed to store frame \(frame.rgbIndex): \(error)")
                        await MainActor.run { [weak self] in
                            self?.error = error
                        }
                    }
                }
                // Signal completion on processing queue
                self.processingQueue.async { [weak self] in
                    self?.pendingStorageTask = nil
                }
            }
        }
    }

    // MARK: - Helpers

    private static func generateSessionName() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmmss"
        return "session_\(formatter.string(from: Date()))"
    }

    /// Resize to 480x360 and encode as JPEG
    private func resizeAndEncodeToJPEG(_ buffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: buffer)

        let scaleX = CGFloat(targetWidth) / ciImage.extent.width
        let scaleY = CGFloat(targetHeight) / ciImage.extent.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        guard let cgImage = ciContext.createCGImage(scaled, from: scaled.extent) else {
            return nil
        }
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: jpegQuality)
    }

    /// Sample the median depth from a 5x5 region at the center of the depth map.
    /// Returns depth in meters, or nil if invalid.
    private func sampleCenterDepth(_ depthMap: CVPixelBuffer) -> Float? {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let centerX = width / 2
        let centerY = height / 2

        var values: [Float] = []
        values.reserveCapacity(25)

        for dy in -2...2 {
            for dx in -2...2 {
                let x = centerX + dx
                let y = centerY + dy
                guard x >= 0, x < width, y >= 0, y < height else { continue }

                let ptr = baseAddress.advanced(by: y * bytesPerRow + x * MemoryLayout<Float>.size)
                let value = ptr.assumingMemoryBound(to: Float.self).pointee

                if value.isFinite && value > 0 {
                    values.append(value)
                }
            }
        }

        guard !values.isEmpty else { return nil }

        // Median
        values.sort()
        return values[values.count / 2]
    }
}
