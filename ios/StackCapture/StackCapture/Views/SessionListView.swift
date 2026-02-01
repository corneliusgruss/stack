import SwiftUI

struct SessionListView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var sessions: [SessionInfo] = []
    @State private var isLoading = true

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView("Loading sessions...")
                } else if sessions.isEmpty {
                    ContentUnavailableView(
                        "No Sessions",
                        systemImage: "folder.badge.questionmark",
                        description: Text("Record your first capture session")
                    )
                } else {
                    List {
                        ForEach(sessions) { session in
                            NavigationLink(destination: SessionDetailView(session: session)) {
                                SessionRowView(session: session)
                            }
                        }
                        .onDelete(perform: deleteSessions)
                    }
                }
            }
            .navigationTitle("Sessions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") {
                        dismiss()
                    }
                }
                if !sessions.isEmpty {
                    ToolbarItem(placement: .topBarTrailing) {
                        EditButton()
                    }
                }
            }
        }
        .onAppear {
            loadSessions()
        }
    }

    private func loadSessions() {
        isLoading = true

        Task {
            let loaded = await Self.loadSessionsFromDisk()
            await MainActor.run {
                sessions = loaded
                isLoading = false
            }
        }
    }

    private func deleteSessions(at offsets: IndexSet) {
        let sessionsToDelete = offsets.map { sessions[$0] }

        for session in sessionsToDelete {
            try? FileManager.default.removeItem(at: session.url)
        }

        sessions.remove(atOffsets: offsets)
    }

    private static func loadSessionsFromDisk() async -> [SessionInfo] {
        let fm = FileManager.default
        guard let documentsURL = fm.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return []
        }

        do {
            let contents = try fm.contentsOfDirectory(
                at: documentsURL,
                includingPropertiesForKeys: [.isDirectoryKey, .totalFileAllocatedSizeKey]
            )

            var sessions: [SessionInfo] = []

            for url in contents {
                guard url.lastPathComponent.hasPrefix("session_") else { continue }

                var isDirectory: ObjCBool = false
                guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory),
                      isDirectory.boolValue else { continue }

                // Load metadata
                let metadataURL = url.appendingPathComponent("metadata.json")
                var metadata: SessionMetadata? = nil

                if fm.fileExists(atPath: metadataURL.path) {
                    do {
                        let data = try Data(contentsOf: metadataURL)
                        let decoder = JSONDecoder()
                        decoder.dateDecodingStrategy = .iso8601
                        metadata = try decoder.decode(SessionMetadata.self, from: data)
                    } catch {
                        print("Failed to load metadata for \(url.lastPathComponent): \(error)")
                    }
                }

                // Calculate folder size
                let size = folderSize(at: url)

                sessions.append(SessionInfo(
                    id: url.lastPathComponent,
                    url: url,
                    metadata: metadata,
                    folderSize: size
                ))
            }

            // Sort by name (most recent first)
            return sessions.sorted { $0.id > $1.id }
        } catch {
            print("Failed to list sessions: \(error)")
            return []
        }
    }

    private static func folderSize(at url: URL) -> Int64 {
        let fm = FileManager.default
        var size: Int64 = 0

        if let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]) {
            for case let fileURL as URL in enumerator {
                if let resourceValues = try? fileURL.resourceValues(forKeys: [.totalFileAllocatedSizeKey]),
                   let fileSize = resourceValues.totalFileAllocatedSize {
                    size += Int64(fileSize)
                }
            }
        }

        return size
    }
}

struct SessionRowView: View {
    let session: SessionInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(session.displayName)
                .font(.headline)

            HStack(spacing: 12) {
                if let meta = session.metadata {
                    Label("\(meta.rgbFrameCount)", systemImage: "photo")
                    Label("\(meta.depthFrameCount)", systemImage: "cube")

                    if let duration = meta.durationSeconds {
                        Label(formatDuration(duration), systemImage: "clock")
                    }
                }

                Spacer()

                Text(session.sizeString)
                    .foregroundColor(.secondary)
            }
            .font(.caption)
            .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }

    private func formatDuration(_ t: Double) -> String {
        let minutes = Int(t) / 60
        let seconds = Int(t) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

#Preview {
    SessionListView()
}
