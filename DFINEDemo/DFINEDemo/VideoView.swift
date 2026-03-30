import PhotosUI
import SwiftUI

struct VideoView: View {
    @StateObject private var processor = VideoProcessor()
    @ObservedObject var detector: ObjectDetector
    @Binding var threshold: Float
    let availableModels: [String]
    @State private var selectedItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                if let frame = processor.currentFrame {
                    Image(uiImage: frame)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .overlay {
                            DetectionOverlayView(
                                detections: filteredDetections,
                                imageSize: processor.videoSize.width > 0
                                    ? processor.videoSize
                                    : frame.size
                            )
                        }
                        .ignoresSafeArea()
                } else {
                    VStack(spacing: 16) {
                        Image(systemName: "video.badge.plus")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("Select a video to detect objects")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(.systemGroupedBackground))
                }

                // Controls overlay
                VStack(spacing: 8) {
                    ModelPickerView(detector: detector, availableModels: availableModels)

                    if processor.isPlaying {
                        ProgressView(value: processor.progress)
                            .tint(.white)
                    }

                    HStack {
                        Text("Confidence")
                            .font(.subheadline)
                        Slider(value: $threshold, in: 0.05...0.95, step: 0.05)
                        Text(String(format: "%.0f%%", threshold * 100))
                            .font(.subheadline)
                            .monospacedDigit()
                            .frame(width: 40)
                    }

                    HStack {
                        if processor.currentFrame != nil {
                            Text("\(filteredDetections.count) detections")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(String(format: "%.1f ms", processor.smoothedInferenceTime))
                                .font(.caption)
                                .monospacedDigit()
                                .foregroundColor(.secondary)
                            Text(String(format: "%.1f fps", processor.smoothedFPS))
                                .font(.caption)
                                .monospacedDigit()
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
            }
            .background(Color.black.ignoresSafeArea())
            .toolbarBackground(.hidden, for: .navigationBar)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    PhotosPicker(selection: $selectedItem, matching: .videos) {
                        Image(systemName: "video.badge.plus")
                    }
                }
            }
            .onAppear {
                processor.setup(detector: detector, threshold: threshold)
            }
            .onDisappear {
                processor.stop()
            }
            .onChange(of: threshold) { _, newValue in
                processor.updateThreshold(newValue)
            }
            .onChange(of: selectedItem) { _, newItem in
                guard let newItem else { return }
                loadAndProcess(item: newItem)
            }
        }
    }

    private var filteredDetections: [Detection] {
        processor.detections.filter { $0.confidence >= threshold }
    }

    private func loadAndProcess(item: PhotosPickerItem) {
        processor.stop()
        Task {
            guard let movie = try? await item.loadTransferable(type: VideoTransferable.self) else { return }
            processor.play(url: movie.url)
        }
    }
}

struct VideoTransferable: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let tempDir = FileManager.default.temporaryDirectory
            let dest = tempDir.appendingPathComponent(received.file.lastPathComponent)
            if FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.copyItem(at: received.file, to: dest)
            return Self(url: dest)
        }
    }
}
