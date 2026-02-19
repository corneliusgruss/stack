import Foundation
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

    /// Called from AVFoundation's thread with each camera pixel buffer for live preview display.
    var onPreviewBuffer: ((CVPixelBuffer) -> Void)?

    // MARK: - Main thread state
    var bleManager: BLEManager?
    var rawCaptureSession: RawCaptureSession?
    private var storageManager: StorageManager?
    private var videoRecorder: VideoRecorder?
    private var sessionURL: URL?
    private var startTime: Date?
    private let rgbResolution: [Int] = [480, 360]

    // MARK: - Processing queue and its state
    private let processingQueue = DispatchQueue(label: "capture.processing", qos: .userInitiated)
    // Only accessed on processingQueue:
    private var firstTimestamp: Double?
    private var rgbIndex: UInt64 = 0
    private var frameBuffer: [FrameData] = []
    private var pendingStorageTask: Task<Void, Never>?

    // Written on main thread, read on processing thread (benign race at boundaries)
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

        self.sessionURL = sessionURL
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
        rawCaptureSession?.startRecordingIMU()

        recordingFlag = true
        isRecording = true
        print("Started capture: \(sessionName)")
    }

    @MainActor
    func stopCapture() async {
        recordingFlag = false
        isRecording = false

        bleManager?.stopRecording()
        rawCaptureSession?.stopRecordingIMU()

        // Wait for processing queue to drain and grab remaining frames + pending task
        let (remaining, pendingTask): ([FrameData], Task<Void, Never>?) = await withCheckedContinuation { continuation in
            processingQueue.async { [weak self] in
                guard let self = self else {
                    continuation.resume(returning: ([], nil))
                    return
                }
                let frames = self.frameBuffer
                let task = self.pendingStorageTask
                self.frameBuffer = []
                self.pendingStorageTask = nil
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

        // Get IMU readings
        let imuReadings = rawCaptureSession?.getRecordedIMU() ?? []

        // Get camera intrinsics
        let intrinsics = rawCaptureSession?.intrinsics

        guard let storage = storageManager, let start = startTime else { return }

        do {
            try await storage.finalize(
                startTime: start,
                rgbResolution: rgbResolution,
                deviceModel: Self.deviceModel,
                iosVersion: UIDevice.current.systemVersion,
                encoderReadings: encoderReadings,
                bleConnected: bleManager?.isConnected ?? false,
                hasVideo: videoRecorder != nil,
                captureSource: .iphoneUltrawide,
                imuReadings: imuReadings,
                cameraIntrinsics: intrinsics
            )
            print("Capture finalized: \(frameCount) RGB frames, \(encoderReadings.count) encoder readings, \(imuReadings.count) IMU readings")

            // Auto-zip for fast Finder/USB transfer
            if let url = sessionURL {
                await zipSession(at: url)
            }
        } catch {
            self.error = error
            print("Failed to finalize: \(error)")
        }

        storageManager = nil
        videoRecorder = nil
        sessionURL = nil
        startTime = nil
    }

    // MARK: - Zip Archive

    private func zipSession(at sessionURL: URL) async {
        let fm = FileManager.default
        let zipURL = sessionURL.deletingLastPathComponent()
            .appendingPathComponent("\(sessionURL.lastPathComponent).zip")

        // Remove existing zip if present
        try? fm.removeItem(at: zipURL)

        let coordinator = NSFileCoordinator()
        var error: NSError?

        coordinator.coordinate(
            readingItemAt: sessionURL,
            options: .forUploading,
            error: &error
        ) { tempURL in
            try? fm.copyItem(at: tempURL, to: zipURL)
        }

        if let error = error {
            print("Failed to create zip: \(error)")
        } else {
            let attrs = try? fm.attributesOfItem(atPath: zipURL.path)
            let size = (attrs?[.size] as? Int64) ?? 0
            let sizeMB = Double(size) / 1_000_000
            print("Session zipped: \(zipURL.lastPathComponent) (\(String(format: "%.1f", sizeMB)) MB)")
        }
    }

    // MARK: - Frame Handling (called from AVFoundation's thread)

    func handleFrame(_ pixelBuffer: CVPixelBuffer, timestamp: Double) {
        // Live preview
        onPreviewBuffer?(pixelBuffer)

        guard recordingFlag else { return }

        // Dispatch heavy work to serial processing queue
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            self.processFrame(timestamp: timestamp, pixelBuffer: pixelBuffer)
        }
    }

    // MARK: - Processing (on processingQueue — serial, no locking needed)

    private func processFrame(timestamp: Double, pixelBuffer: CVPixelBuffer) {
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

        let frameData = FrameData(
            timestamp: timestamp,
            rgbIndex: currentIndex,
            depth: nil,
            transform: nil,  // Poses filled by SLAM later
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
                let queue = self.processingQueue
                queue.async { [weak self] in
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

    static var deviceModel: String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machineMirror = Mirror(reflecting: systemInfo.machine)
        let identifier = machineMirror.children.reduce("") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return identifier }
            return identifier + String(UnicodeScalar(UInt8(value)))
        }
        return identifier
    }
}
