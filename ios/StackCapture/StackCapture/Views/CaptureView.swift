import SwiftUI
import ARKit

struct CaptureView: View {
    @StateObject private var arManager = ARSessionManager()
    @StateObject private var captureCoordinator = CaptureCoordinator()
    @State private var showingSessions = false
    @State private var arCoordinator: ARSessionCoordinator?

    var body: some View {
        ZStack {
            // AR Preview
            ARViewContainer(arView: arManager.arView)
                .ignoresSafeArea()

            // Overlay
            VStack {
                // Top bar
                HStack {
                    // Tracking status
                    TrackingStatusView(state: arManager.trackingState)

                    Spacer()

                    // Sessions button
                    Button {
                        showingSessions = true
                    } label: {
                        Image(systemName: "folder.fill")
                            .font(.title2)
                            .foregroundColor(.white)
                            .padding(12)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                }
                .padding()

                Spacer()

                // Recording stats
                if captureCoordinator.isRecording {
                    RecordingStatsView(
                        duration: captureCoordinator.duration,
                        frameCount: captureCoordinator.frameCount,
                        depthCount: captureCoordinator.depthFrameCount
                    )
                    .padding(.bottom, 20)
                }

                // Record button
                RecordButton(isRecording: captureCoordinator.isRecording) {
                    toggleRecording()
                }
                .padding(.bottom, 40)
            }
        }
        .onAppear {
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

// MARK: - Subviews

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
        case .normal:
            return .green
        case .limited:
            return .yellow
        case .notAvailable:
            return .red
        }
    }

    private var statusText: String {
        switch state {
        case .normal:
            return "Tracking"
        case .limited(let reason):
            switch reason {
            case .excessiveMotion:
                return "Too fast"
            case .insufficientFeatures:
                return "Low features"
            case .initializing:
                return "Initializing"
            case .relocalizing:
                return "Relocalizing"
            @unknown default:
                return "Limited"
            }
        case .notAvailable:
            return "Not available"
        }
    }
}

struct RecordingStatsView: View {
    let duration: TimeInterval
    let frameCount: UInt64
    let depthCount: UInt64

    var body: some View {
        HStack(spacing: 20) {
            StatItem(icon: "clock", value: formatDuration(duration))
            StatItem(icon: "photo", value: "\(frameCount)")
            StatItem(icon: "cube", value: "\(depthCount)")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
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
                    .frame(width: 80, height: 80)

                if isRecording {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.red)
                        .frame(width: 32, height: 32)
                } else {
                    Circle()
                        .fill(.red)
                        .frame(width: 64, height: 64)
                }
            }
        }
    }
}

#Preview {
    CaptureView()
}
