import SwiftUI
import AVFoundation
import CoreMedia

struct CaptureView: View {
    @StateObject private var captureCoordinator = CaptureCoordinator()
    @StateObject private var bleManager = BLEManager()
    @StateObject private var rawSession = RawCaptureSession()
    @State private var showingSessions = false

    var body: some View {
        ZStack {
            // Camera preview (full screen)
            PixelBufferPreviewView(coordinator: captureCoordinator)
                .ignoresSafeArea()

            // Landscape overlay
            VStack(spacing: 0) {
                // Top bar
                HStack(alignment: .top) {
                    // Top-left: camera status + BLE status
                    VStack(alignment: .leading, spacing: 6) {
                        CameraStatusView(status: rawSession.cameraStatus)
                        BLEStatusView(state: bleManager.connectionState)
                    }

                    Spacer()

                    // Top-right: encoder values + sessions button
                    HStack(spacing: 12) {
                        EncoderValuesView(
                            values: bleManager.encoderValues,
                            hasError: bleManager.hasError
                        )

                        Button {
                            showingSessions = true
                        } label: {
                            Image(systemName: "folder.fill")
                                .font(.title3)
                                .foregroundColor(.white)
                                .padding(10)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)

                Spacer()

                // Bottom bar
                HStack(alignment: .bottom) {
                    // Bottom-left: recording stats
                    if captureCoordinator.isRecording {
                        RecordingStatsView(
                            duration: captureCoordinator.duration,
                            frameCount: captureCoordinator.frameCount
                        )
                    } else {
                        Spacer().frame(width: 160)
                    }

                    Spacer()

                    // Bottom-center: record button
                    RecordButton(isRecording: captureCoordinator.isRecording) {
                        toggleRecording()
                    }

                    Spacer()

                    // Bottom-right: calibrate button
                    if bleManager.isConnected {
                        Button {
                            bleManager.calibrate()
                        } label: {
                            VStack(spacing: 2) {
                                Image(systemName: "scope")
                                    .font(.title3)
                                Text("Zero")
                                    .font(.caption2)
                            }
                            .foregroundColor(.white)
                            .padding(10)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
                        }
                        .frame(width: 160, alignment: .trailing)
                    } else {
                        Spacer().frame(width: 160)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 16)
            }
        }
        .onAppear {
            captureCoordinator.bleManager = bleManager
            captureCoordinator.rawCaptureSession = rawSession
            rawSession.onFrame = { pixelBuffer, timestamp in
                captureCoordinator.handleFrame(pixelBuffer, timestamp: timestamp)
            }
            rawSession.startSession()
        }
        .sheet(isPresented: $showingSessions) {
            SessionListView()
        }
        .alert("Error", isPresented: .constant(captureCoordinator.error != nil)) {
            Button("OK") {
                captureCoordinator.error = nil
            }
        } message: {
            if let error = captureCoordinator.error {
                Text(error.localizedDescription)
            }
        }
    }

    private func toggleRecording() {
        if captureCoordinator.isRecording {
            Task {
                await captureCoordinator.stopCapture()
            }
        } else {
            do {
                try captureCoordinator.startCapture()
            } catch {
                captureCoordinator.error = error
            }
        }
    }
}

// MARK: - Camera Status

struct CameraStatusView: View {
    let status: RawCaptureSession.CameraStatus

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(status.rawValue)
                .font(.caption)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var statusColor: Color {
        switch status {
        case .running: return .green
        case .notStarted: return .yellow
        case .failed: return .red
        }
    }
}

// MARK: - Pixel Buffer Preview

struct PixelBufferPreviewView: UIViewRepresentable {
    let coordinator: CaptureCoordinator

    func makeUIView(context: Context) -> PreviewDisplayView {
        let view = PreviewDisplayView()
        coordinator.onPreviewBuffer = { [weak view] pixelBuffer in
            view?.displayPixelBuffer(pixelBuffer)
        }
        return view
    }

    func updateUIView(_ uiView: PreviewDisplayView, context: Context) {}
}

class PreviewDisplayView: UIView {
    private var displayLayer: AVSampleBufferDisplayLayer!

    override init(frame: CGRect) {
        super.init(frame: frame)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        displayLayer = AVSampleBufferDisplayLayer()
        displayLayer.videoGravity = .resizeAspectFill
        layer.addSublayer(displayLayer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        displayLayer.frame = bounds
    }

    func displayPixelBuffer(_ pixelBuffer: CVPixelBuffer) {
        guard let sampleBuffer = Self.makeSampleBuffer(from: pixelBuffer) else { return }
        DispatchQueue.main.async { [weak self] in
            guard let layer = self?.displayLayer else { return }
            if layer.status == .failed {
                layer.flush()
            }
            layer.enqueue(sampleBuffer)
        }
    }

    private static func makeSampleBuffer(from pixelBuffer: CVPixelBuffer) -> CMSampleBuffer? {
        var formatDesc: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: nil,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDesc
        )
        guard let format = formatDesc else { return nil }

        var timingInfo = CMSampleTimingInfo(
            duration: .invalid,
            presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
            decodeTimeStamp: .invalid
        )

        var sampleBuffer: CMSampleBuffer?
        CMSampleBufferCreateForImageBuffer(
            allocator: nil,
            imageBuffer: pixelBuffer,
            dataReady: true,
            makeDataReadyCallback: nil,
            refcon: nil,
            formatDescription: format,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )

        return sampleBuffer
    }
}

// MARK: - BLE Status

struct BLEStatusView: View {
    let state: BLEManager.ConnectionState

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "hand.raised.fill")
                .font(.caption2)
            Text(state.rawValue)
                .font(.caption)
        }
        .foregroundColor(stateColor)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var stateColor: Color {
        switch state {
        case .connected: return .green
        case .scanning, .connecting: return .orange
        case .disconnected: return .red
        }
    }
}

// MARK: - Encoder Values Display

struct EncoderValuesView: View {
    let values: [Float]
    let hasError: Bool

    private let labels = ["iMCP", "iPIP", "3MCP", "3PIP"]

    var body: some View {
        HStack(spacing: 8) {
            ForEach(0..<4, id: \.self) { i in
                VStack(spacing: 1) {
                    Text(labels[i])
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundColor(.white.opacity(0.7))
                    Text(values[i] < 0 ? "ERR" : String(format: "%.0f", values[i]))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(values[i] < 0 ? .red : .white)
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Recording Stats

struct RecordingStatsView: View {
    let duration: TimeInterval
    let frameCount: UInt64

    var body: some View {
        HStack(spacing: 16) {
            StatItem(icon: "clock", value: formatDuration(duration))
            StatItem(icon: "photo", value: "\(frameCount)")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func formatDuration(_ t: TimeInterval) -> String {
        let minutes = Int(t) / 60
        let seconds = Int(t) % 60
        let tenths = Int((t - Double(Int(t))) * 10)
        return String(format: "%d:%02d.%d", minutes, seconds, tenths)
    }
}

struct StatItem: View {
    let icon: String
    let value: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
            Text(value)
                .font(.system(.body, design: .monospaced))
        }
        .foregroundColor(.white)
    }
}

struct RecordButton: View {
    let isRecording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .stroke(.white, lineWidth: 4)
                    .frame(width: 72, height: 72)

                if isRecording {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.red)
                        .frame(width: 28, height: 28)
                } else {
                    Circle()
                        .fill(.red)
                        .frame(width: 56, height: 56)
                }
            }
        }
    }
}

#Preview {
    CaptureView()
}
