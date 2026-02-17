import AVFoundation
import CoreVideo
import VideoToolbox

/// Records full-resolution HEVC video from ARKit pixel buffers using AVAssetWriter.
/// Hardware-accelerated on iPhone 16 Pro — essentially free CPU.
class VideoRecorder {
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var startTime: CMTime?
    private var frameCount: UInt64 = 0
    private let outputURL: URL

    init(outputURL: URL) {
        self.outputURL = outputURL
    }

    /// Start the video writer. Call before appending frames.
    func start(width: Int, height: Int) throws {
        // Remove existing file
        try? FileManager.default.removeItem(at: outputURL)

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 8_000_000,  // 8 Mbps — good quality for 60fps
                AVVideoExpectedSourceFrameRateKey: 60,
                AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main_AutoLevel,
            ]
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = true

        let sourcePixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: sourcePixelBufferAttributes
        )

        guard writer.canAdd(input) else {
            throw VideoRecorderError.cannotAddInput
        }

        writer.add(input)

        guard writer.startWriting() else {
            throw writer.error ?? VideoRecorderError.startFailed
        }

        writer.startSession(atSourceTime: .zero)

        self.assetWriter = writer
        self.videoInput = input
        self.pixelBufferAdaptor = adaptor
        self.startTime = nil
        self.frameCount = 0

        print("VideoRecorder started: \(width)x\(height) HEVC → \(outputURL.lastPathComponent)")
    }

    /// Append a pixel buffer from ARKit. Thread-safe — call from any thread.
    func appendFrame(_ pixelBuffer: CVPixelBuffer, timestamp: Double) {
        guard let input = videoInput, input.isReadyForMoreMediaData else { return }

        let presentationTime: CMTime
        if let start = startTime {
            let elapsed = timestamp - CMTimeGetSeconds(start)
            presentationTime = CMTimeMakeWithSeconds(elapsed, preferredTimescale: 600)
        } else {
            startTime = CMTimeMakeWithSeconds(timestamp, preferredTimescale: 600)
            presentationTime = .zero
        }

        pixelBufferAdaptor?.append(pixelBuffer, withPresentationTime: presentationTime)
        frameCount += 1
    }

    /// Finalize the video file. Must be called before the file is usable.
    func finish() async {
        guard let writer = assetWriter else { return }

        videoInput?.markAsFinished()

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting {
                continuation.resume()
            }
        }

        let fileSize = (try? FileManager.default.attributesOfItem(atPath: outputURL.path)[.size] as? Int64) ?? 0
        let sizeMB = Double(fileSize) / 1_000_000
        print("VideoRecorder finished: \(frameCount) frames, \(String(format: "%.1f", sizeMB)) MB")

        assetWriter = nil
        videoInput = nil
        pixelBufferAdaptor = nil
    }
}

enum VideoRecorderError: Error {
    case cannotAddInput
    case startFailed
}
