import Foundation
import ARKit
import Combine
import UIKit
import CoreImage

// MARK: - Capture Coordinator

@MainActor
class CaptureCoordinator: ObservableObject {
    @Published var isRecording = false
    @Published var frameCount: UInt64 = 0
    @Published var depthFrameCount: UInt64 = 0
    @Published var duration: TimeInterval = 0
    @Published var error: Error?

    private var storageManager: StorageManager?
    private var startTime: Date?
    private var firstTimestamp: Double?
    private var lastDepthTime: Double = 0
    private var rgbIndex: UInt64 = 0
    private var depthIndex: UInt64 = 0

    private var rgbResolution: [Int] = [1920, 1440]
    private var depthResolution: [Int] = [256, 192]

    private var frameBuffer: [FrameData] = []
    private var isProcessing = false

    // Depth capture at 30Hz (every other frame at 60fps)
    private let depthInterval: Double = 1.0 / 30.0
    private let jpegQuality: CGFloat = 0.85
    private let ciContext = CIContext()

    // MARK: - Public API

    func startCapture() throws {
        // Create session directory
        let sessionName = Self.generateSessionName()
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let sessionURL = documentsURL.appendingPathComponent(sessionName)

        storageManager = try StorageManager(sessionURL: sessionURL)

        // Reset state
        startTime = Date()
        firstTimestamp = nil
        lastDepthTime = 0
        rgbIndex = 0
        depthIndex = 0
        frameCount = 0
        depthFrameCount = 0
        duration = 0
        error = nil
        frameBuffer = []

        isRecording = true
        print("Started capture: \(sessionName)")
    }

    func stopCapture() async {
        isRecording = false

        // Process remaining frames
        await processBufferedFrames()

        // Finalize storage
        guard let storage = storageManager, let start = startTime else { return }

        do {
            try await storage.finalize(
                startTime: start,
                rgbResolution: rgbResolution,
                depthResolution: depthResolution,
                deviceModel: ARSessionManager.deviceModel,
                iosVersion: UIDevice.current.systemVersion
            )
            print("Capture finalized: \(frameCount) RGB, \(depthFrameCount) depth frames")
        } catch {
            self.error = error
            print("Failed to finalize: \(error)")
        }

        storageManager = nil
        startTime = nil
    }

    // MARK: - Frame Handling

    func handleFrame(_ frame: ARFrame) {
        guard isRecording else { return }

        let timestamp = frame.timestamp

        // Track first timestamp for duration
        if firstTimestamp == nil {
            firstTimestamp = timestamp
        }

        // Update duration
        if let first = firstTimestamp {
            duration = timestamp - first
        }

        // Determine if we capture depth this frame
        var captureDepth = false
        var currentDepthIndex: UInt64? = nil

        if timestamp - lastDepthTime >= depthInterval {
            if frame.smoothedSceneDepth?.depthMap != nil {
                captureDepth = true
                currentDepthIndex = depthIndex
                lastDepthTime = timestamp
            }
        }

        // Get resolution from first frame
        if rgbIndex == 0 {
            let rgbBuffer = frame.capturedImage
            rgbResolution = [CVPixelBufferGetWidth(rgbBuffer), CVPixelBufferGetHeight(rgbBuffer)]

            if let depthBuffer = frame.smoothedSceneDepth?.depthMap {
                depthResolution = [CVPixelBufferGetWidth(depthBuffer), CVPixelBufferGetHeight(depthBuffer)]
            }
        }

        // Encode RGB to JPEG immediately (ARKit recycles buffers)
        guard let jpegData = encodeToJPEG(frame.capturedImage) else {
            print("Failed to encode frame \(rgbIndex) to JPEG")
            return
        }

        // Copy depth data immediately
        var depthData: Data? = nil
        if captureDepth, let depthBuffer = frame.smoothedSceneDepth?.depthMap {
            depthData = copyDepthBuffer(depthBuffer)
        }

        // Create frame data with copied/encoded data
        let frameData = FrameData(
            timestamp: timestamp,
            rgbIndex: rgbIndex,
            depthIndex: captureDepth ? currentDepthIndex : nil,
            transform: frame.camera.transform,
            jpegData: jpegData,
            depthData: depthData
        )

        // Update indices
        rgbIndex += 1
        if captureDepth {
            depthIndex += 1
        }

        // Update UI counters
        frameCount = rgbIndex
        depthFrameCount = depthIndex

        // Queue for async processing
        queueFrame(frameData)
    }

    // MARK: - Async Processing

    private func queueFrame(_ data: FrameData) {
        frameBuffer.append(data)

        // Process in batches to avoid overwhelming storage
        if frameBuffer.count >= 10 && !isProcessing {
            Task {
                await processBufferedFrames()
            }
        }
    }

    private func processBufferedFrames() async {
        guard !frameBuffer.isEmpty else { return }

        isProcessing = true
        let framesToProcess = frameBuffer
        frameBuffer = []

        guard let storage = storageManager else {
            isProcessing = false
            return
        }

        for frame in framesToProcess {
            do {
                try await storage.storeFrame(frame)
            } catch {
                print("Failed to store frame \(frame.rgbIndex): \(error)")
                self.error = error
            }
        }

        isProcessing = false
    }

    // MARK: - Helpers

    private static func generateSessionName() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmmss"
        return "session_\(formatter.string(from: Date()))"
    }

    private func encodeToJPEG(_ buffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: jpegQuality)
    }

    private func copyDepthBuffer(_ buffer: CVPixelBuffer) -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let baseAddress = CVPixelBufferGetBaseAddress(buffer)!

        return Data(bytes: baseAddress, count: height * bytesPerRow)
    }
}
