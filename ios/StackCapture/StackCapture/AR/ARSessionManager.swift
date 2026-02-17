import SwiftUI
import ARKit

// MARK: - AR Session Manager

@MainActor
class ARSessionManager: ObservableObject {
    let session = ARSession()
    private var coordinator: ARSessionCoordinator?

    @Published var isSessionRunning = false
    @Published var trackingState: ARCamera.TrackingState = .notAvailable

    func startSession(coordinator: ARSessionCoordinator) {
        self.coordinator = coordinator

        let configuration = ARWorldTrackingConfiguration()

        // Enable scene depth (LiDAR) — keeps LiDAR-assisted tracking even though we don't save full depth maps
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
        }

        // Select 1920x1440 @ 60fps (good balance for video + training)
        // ARKit doesn't support ultrawide camera for world tracking, so we use wide-angle
        let allFormats = ARWorldTrackingConfiguration.supportedVideoFormats
        let selectedFormat = allFormats.first(where: {
            $0.framesPerSecond == 60 && $0.imageResolution.width == 1920 && $0.imageResolution.height == 1440
        }) ?? allFormats.first(where: { $0.framesPerSecond == 60 })

        if let format = selectedFormat {
            configuration.videoFormat = format
            let res = format.imageResolution
            print("Selected format: \(Int(res.width))x\(Int(res.height)) @ \(format.framesPerSecond) fps")
        }

        session.delegate = coordinator
        session.run(configuration)
        isSessionRunning = true
    }

    func pauseSession() {
        session.pause()
        isSessionRunning = false
    }

    func resumeSession() {
        guard let config = session.configuration else { return }
        session.run(config)
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
