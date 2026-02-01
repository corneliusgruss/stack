import ARKit
import Combine

// MARK: - AR Session Coordinator

class ARSessionCoordinator: NSObject, ARSessionDelegate {
    private let onFrame: (ARFrame) -> Void
    private let onTrackingStateChange: (ARCamera.TrackingState) -> Void

    init(
        onFrame: @escaping (ARFrame) -> Void,
        onTrackingStateChange: @escaping (ARCamera.TrackingState) -> Void
    ) {
        self.onFrame = onFrame
        self.onTrackingStateChange = onTrackingStateChange
        super.init()
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        onFrame(frame)
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        onTrackingStateChange(camera.trackingState)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("ARSession failed: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        print("ARSession interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        print("ARSession interruption ended")
    }
}
