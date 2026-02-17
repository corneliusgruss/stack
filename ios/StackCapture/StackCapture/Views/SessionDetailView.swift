import SwiftUI

struct SessionDetailView: View {
    let session: SessionInfo

    @State private var isExporting = false
    @State private var shareURL: URL?
    @State private var exportError: Error?

    var body: some View {
        List {
            Section("Session Info") {
                LabeledContent("Name", value: session.id)
                LabeledContent("Size", value: session.sizeString)

                if let meta = session.metadata {
                    LabeledContent("Device", value: meta.deviceModel)
                    LabeledContent("iOS Version", value: meta.iosVersion)

                    if let duration = meta.durationSeconds {
                        LabeledContent("Duration", value: formatDuration(duration))
                    }
                }
            }

            if let meta = session.metadata {
                Section("Capture Stats") {
                    LabeledContent("RGB Frames", value: "\(meta.rgbFrameCount)")
                    LabeledContent("Pose Samples", value: "\(meta.poseCount)")
                    LabeledContent("RGB Resolution", value: "\(meta.rgbResolution[0])x\(meta.rgbResolution[1])")

                    if let encoderCount = meta.encoderCount {
                        LabeledContent("Encoder Readings", value: "\(encoderCount)")
                    }
                    if let ble = meta.bleConnected {
                        LabeledContent("Glove Connected", value: ble ? "Yes" : "No")
                    }
                    if let video = meta.hasVideo {
                        LabeledContent("Video Recorded", value: video ? "Yes" : "No")
                    }
                }

                Section("Timestamps") {
                    LabeledContent("Started", value: formatDate(meta.startTime))
                    if let end = meta.endTime {
                        LabeledContent("Ended", value: formatDate(end))
                    }
                }
            }

            Section {
                Button {
                    exportSession()
                } label: {
                    HStack {
                        if isExporting {
                            ProgressView()
                                .padding(.trailing, 8)
                        }
                        Image(systemName: "square.and.arrow.up")
                        Text(isExporting ? "Preparing..." : "Export Session")
                    }
                }
                .disabled(isExporting)
            }
        }
        .navigationTitle("Session Details")
        .sheet(item: $shareURL) { url in
            ShareSheet(url: url)
        }
        .alert("Export Failed", isPresented: .constant(exportError != nil)) {
            Button("OK") { exportError = nil }
        } message: {
            if let error = exportError {
                Text(error.localizedDescription)
            }
        }
    }

    private func formatDuration(_ t: Double) -> String {
        let minutes = Int(t) / 60
        let seconds = Int(t) % 60
        let tenths = Int((t - Double(Int(t))) * 10)
        return String(format: "%d:%02d.%d", minutes, seconds, tenths)
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .medium
        return formatter.string(from: date)
    }

    private func exportSession() {
        isExporting = true
        exportError = nil

        Task {
            do {
                let zipURL = try await createZipArchive()
                await MainActor.run {
                    isExporting = false
                    shareURL = zipURL
                }
            } catch {
                await MainActor.run {
                    isExporting = false
                    exportError = error
                }
            }
        }
    }

    private func createZipArchive() async throws -> URL {
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory
        let zipName = "\(session.id).zip"
        let zipURL = tempDir.appendingPathComponent(zipName)

        // Remove existing zip if present
        try? fm.removeItem(at: zipURL)

        // Create zip using Foundation's built-in compression
        let coordinator = NSFileCoordinator()
        var error: NSError?

        coordinator.coordinate(
            readingItemAt: session.url,
            options: .forUploading,
            error: &error
        ) { tempURL in
            try? fm.copyItem(at: tempURL, to: zipURL)
        }

        if let error = error {
            throw error
        }

        return zipURL
    }
}

// MARK: - URL Identifiable Extension

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}

#Preview {
    NavigationStack {
        SessionDetailView(session: SessionInfo(
            id: "session_2026-02-16_153045",
            url: URL(fileURLWithPath: "/tmp"),
            metadata: SessionMetadata(
                deviceModel: "iPhone16,1",
                iosVersion: "18.3",
                startTime: Date(),
                endTime: Date().addingTimeInterval(30.5),
                rgbResolution: [480, 360],
                rgbFrameCount: 1830,
                poseCount: 1830,
                durationSeconds: 30.5,
                encoderCount: 1525,
                bleConnected: true,
                hasVideo: true
            ),
            folderSize: 34_000_000
        ))
    }
}
