import SwiftUI
import ARKit
import RealityKit

// MARK: - AR View Container

struct ARViewContainer: UIViewRepresentable {
    let arView: ARView

    func makeUIView(context: Context) -> ARView {
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

// MARK: - AR Session Manager

@MainActor
class ARSessionManager: ObservableObject {
    let arView: ARView
    private var coordinator: ARSessionCoordinator?

    @Published var isSessionRunning = false
    @Published var trackingState: ARCamera.TrackingState = .notAvailable

    init() {
        arView = ARView(frame: .zero)
        arView.automaticallyConfigureSession = false
    }

    func startSession(coordinator: ARSessionCoordinator) {
        self.coordinator = coordinator

        let configuration = ARWorldTrackingConfiguration()

        // Enable scene depth (LiDAR)
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
        }

        // Request highest frame rate
        if let videoFormat = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: { $0.framesPerSecond == 60 }) {
            configuration.videoFormat = videoFormat
        }

        arView.session.delegate = coordinator
        arView.session.run(configuration)
        isSessionRunning = true
    }

    func pauseSession() {
        arView.session.pause()
        isSessionRunning = false
    }

    func resumeSession() {
        guard let config = arView.session.configuration else { return }
        arView.session.run(config)
        isSessionRunning = true
    }

    static var supportsLiDAR: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
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
