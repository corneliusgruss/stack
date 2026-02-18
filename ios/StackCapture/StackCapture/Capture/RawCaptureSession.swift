import AVFoundation
import CoreMotion
import CoreVideo
import UIKit

/// AVFoundation-based capture using the ultrawide camera (no ARKit).
/// Captures RGB frames, IMU data, and camera intrinsics.
/// Poses are left empty — filled later by DROID-SLAM offline processing.
class RawCaptureSession: NSObject, ObservableObject {
    // MARK: - Published state

    @Published var isRunning = false
    @Published var cameraStatus: CameraStatus = .notStarted

    enum CameraStatus: String {
        case notStarted = "Not Started"
        case running = "Running"
        case failed = "Failed"
    }

    /// Called with each camera pixel buffer for live preview display.
    var onFrame: ((CVPixelBuffer, Double) -> Void)?

    // MARK: - AVFoundation

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "raw.capture.session", qos: .userInitiated)

    // MARK: - IMU

    private let motionManager = CMMotionManager()
    private var imuReadings: [IMUReading] = []
    private let imuLock = NSLock()
    private var isRecordingIMU = false

    // MARK: - Intrinsics

    private(set) var intrinsics: CameraIntrinsics?
    private var nativeResolution: CGSize = .zero

    // MARK: - Setup

    func startSession() {
        sessionQueue.async { [weak self] in
            self?.configureSession()
        }
    }

    func stopSession() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
            self?.motionManager.stopDeviceMotionUpdates()
            DispatchQueue.main.async {
                self?.isRunning = false
                self?.cameraStatus = .notStarted
            }
        }
    }

    // MARK: - IMU Recording

    func startRecordingIMU() {
        imuLock.lock()
        imuReadings = []
        isRecordingIMU = true
        imuLock.unlock()
    }

    func stopRecordingIMU() {
        imuLock.lock()
        isRecordingIMU = false
        imuLock.unlock()
    }

    func getRecordedIMU() -> [IMUReading] {
        imuLock.lock()
        let readings = imuReadings
        imuLock.unlock()
        return readings
    }

    // MARK: - Private

    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .inputPriority

        // Find ultrawide camera (13mm, ~120° FOV)
        guard let device = AVCaptureDevice.default(
            .builtInUltraWideCamera,
            for: .video,
            position: .back
        ) else {
            print("RawCapture: Ultrawide camera not available")
            DispatchQueue.main.async { self.cameraStatus = .failed }
            captureSession.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard captureSession.canAddInput(input) else {
                print("RawCapture: Cannot add camera input")
                captureSession.commitConfiguration()
                return
            }
            captureSession.addInput(input)

            // Select best format: prefer 1920x1440 @ 60fps, fallback to 30fps
            let targetWidth: Int32 = 1920
            let targetHeight: Int32 = 1440
            let formats = device.formats.filter { format in
                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                return dims.width == targetWidth && dims.height == targetHeight
            }

            let selectedFormat = formats.first(where: { format in
                format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 60 }
            }) ?? formats.first ?? device.activeFormat

            try device.lockForConfiguration()
            device.activeFormat = selectedFormat
            let maxFPS = selectedFormat.videoSupportedFrameRateRanges
                .max(by: { $0.maxFrameRate < $1.maxFrameRate })?.maxFrameRate ?? 30
            let fps = min(maxFPS, 60)
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.unlockForConfiguration()

            let dims = CMVideoFormatDescriptionGetDimensions(selectedFormat.formatDescription)
            nativeResolution = CGSize(width: CGFloat(dims.width), height: CGFloat(dims.height))
            print("RawCapture: Selected \(dims.width)x\(dims.height) @ \(Int(fps))fps ultrawide")

        } catch {
            print("RawCapture: Camera setup failed: \(error)")
            DispatchQueue.main.async { self.cameraStatus = .failed }
            captureSession.commitConfiguration()
            return
        }

        // Video output
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]

        guard captureSession.canAddOutput(videoOutput) else {
            print("RawCapture: Cannot add video output")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addOutput(videoOutput)

        // Enable camera intrinsic matrix delivery on the video connection
        if let connection = videoOutput.connection(with: .video) {
            if connection.isCameraIntrinsicMatrixDeliverySupported {
                connection.isCameraIntrinsicMatrixDeliveryEnabled = true
                print("RawCapture: Intrinsic matrix delivery enabled")
            } else {
                print("RawCapture: Intrinsic matrix delivery not supported — DROID-SLAM will auto-calibrate")
            }
        }

        captureSession.commitConfiguration()
        captureSession.startRunning()

        // Start IMU at 200 Hz
        startIMU()

        DispatchQueue.main.async {
            self.isRunning = true
            self.cameraStatus = .running
        }
    }

    /// Extract intrinsics from sample buffer attachment (called once on first frame).
    private func extractIntrinsics(from sampleBuffer: CMSampleBuffer) {
        guard intrinsics == nil else { return }  // Only extract once

        // Read kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix from buffer
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[String: Any]],
           let first = attachments.first,
           let matrixData = first[kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix as String] as? Data {
            var matrix = matrix_float3x3()
            _ = matrixData.withUnsafeBytes { ptr in
                memcpy(&matrix, ptr.baseAddress!, MemoryLayout<matrix_float3x3>.size)
            }

            let fx = matrix.columns.0.x
            let fy = matrix.columns.1.y
            let cx = matrix.columns.2.x
            let cy = matrix.columns.2.y

            // Scale from native resolution to 480x360
            let scaleX = Float(480) / Float(nativeResolution.width)
            let scaleY = Float(360) / Float(nativeResolution.height)

            self.intrinsics = CameraIntrinsics(
                fx: fx * scaleX,
                fy: fy * scaleY,
                cx: cx * scaleX,
                cy: cy * scaleY
            )
            print("RawCapture: Intrinsics (scaled to 480x360): fx=\(fx * scaleX), fy=\(fy * scaleY), cx=\(cx * scaleX), cy=\(cy * scaleY)")
        }
    }

    private func startIMU() {
        guard motionManager.isDeviceMotionAvailable else {
            print("RawCapture: Device motion not available")
            return
        }

        motionManager.deviceMotionUpdateInterval = 1.0 / 200.0  // 200 Hz
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self = self, let motion = motion else { return }

            let reading = IMUReading(
                timestamp: Date().timeIntervalSince1970,
                accel: [motion.userAcceleration.x, motion.userAcceleration.y, motion.userAcceleration.z],
                gyro: [motion.rotationRate.x, motion.rotationRate.y, motion.rotationRate.z]
            )

            self.imuLock.lock()
            if self.isRecordingIMU {
                self.imuReadings.append(reading)
            }
            self.imuLock.unlock()
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension RawCaptureSession: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Extract intrinsics from first frame
        extractIntrinsics(from: sampleBuffer)

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        onFrame?(pixelBuffer, timestamp)
    }
}
