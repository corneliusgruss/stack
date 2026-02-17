import SwiftUI
import ARKit
import AVFoundation
import CoreMedia

struct CaptureView: View {
    @StateObject private var arManager = ARSessionManager()
    @StateObject private var captureCoordinator = CaptureCoordinator()
    @StateObject private var bleManager = BLEManager()
    @State private var showingSessions = false
    @State private var arCoordinator: ARSessionCoordinator?

    var body: some View {
        ZStack {
            // Camera preview (full screen) — lightweight AVSampleBufferDisplayLayer, no RealityKit
            PixelBufferPreviewView(coordinator: captureCoordinator)
                .ignoresSafeArea()

            // Depth crosshair at center
            DepthCrosshair()

            // Landscape overlay
            VStack(spacing: 0) {
                // Top bar
                HStack(alignment: .top) {
                    // Top-left: tracking status + BLE status
                    VStack(alignment: .leading, spacing: 6) {
                        TrackingStatusView(state: arManager.trackingState)
                        BLEStatusView(state: bleManager.connectionState)
                    }

                    Spacer()

                    // Top-right: encoder values + depth + sessions button
                    HStack(spacing: 12) {
                        EncoderValuesView(
                            values: bleManager.encoderValues,
                            hasError: bleManager.hasError,
                            depthPoint: captureCoordinator.currentDepthPoint
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
            setupARSession()
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

    private func setupARSession() {
        let coordinator = ARSessionCoordinator(
            onFrame: { frame in
                captureCoordinator.handleFrame(frame)
            },
            onTrackingStateChange: { state in
                Task { @MainActor in
                    arManager.trackingState = state
                }
            }
        )
        arCoordinator = coordinator
        arManager.startSession(coordinator: coordinator)
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

// MARK: - Pixel Buffer Preview (replaces ARView/RealityKit)

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

    /// Display a pixel buffer from ARKit. Can be called from any thread.
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

// MARK: - Depth Crosshair

struct DepthCrosshair: View {
    var body: some View {
        ZStack {
            // Horizontal line
            Rectangle()
                .fill(.white.opacity(0.6))
                .frame(width: 20, height: 1)
            // Vertical line
            Rectangle()
                .fill(.white.opacity(0.6))
                .frame(width: 1, height: 20)
            // Center dot
            Circle()
                .stroke(.white.opacity(0.6), lineWidth: 1)
                .frame(width: 6, height: 6)
        }
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
    let depthPoint: Float?

    private let labels = ["iMCP", "iPIP", "3MCP", "3PIP"]

    var body: some View {
        HStack(spacing: 8) {
            // Encoder angles
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

            // Depth point
            if let depth = depthPoint {
                Divider()
                    .frame(height: 24)
                    .background(.white.opacity(0.3))
                VStack(spacing: 1) {
                    Text("Depth")
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundColor(.white.opacity(0.7))
                    Text(String(format: "%.2fm", depth))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(depthColor(depth))
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private func depthColor(_ d: Float) -> Color {
        if d.isNaN || d <= 0 { return .red }
        if d >= 0.1 && d <= 1.0 { return .green }
        return .yellow
    }
}

// MARK: - Tracking Status

struct TrackingStatusView: View {
    let state: ARCamera.TrackingState

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.caption)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var statusColor: Color {
        switch state {
        case .normal: return .green
        case .limited: return .yellow
        case .notAvailable: return .red
        }
    }

    private var statusText: String {
        switch state {
        case .normal:
            return "Tracking"
        case .limited(let reason):
            switch reason {
            case .excessiveMotion: return "Too fast"
            case .insufficientFeatures: return "Low features"
            case .initializing: return "Initializing"
            case .relocalizing: return "Relocalizing"
            @unknown default: return "Limited"
            }
        case .notAvailable:
            return "Not available"
        }
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
