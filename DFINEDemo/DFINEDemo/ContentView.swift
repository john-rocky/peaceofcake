import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject private var detector = ObjectDetector()
    @State private var selectedTab = 0
    @State private var threshold: Float = 0.5
    @State private var availableModels: [String] = ObjectDetector.availableModels()

    var body: some View {
        TabView(selection: $selectedTab) {
            PhotoDetectionView(detector: detector, threshold: $threshold, availableModels: availableModels)
                .tabItem {
                    Label("Photo", systemImage: "photo")
                }
                .tag(0)

            CameraView(detector: detector, threshold: $threshold, availableModels: availableModels)
                .tabItem {
                    Label("Camera", systemImage: "camera")
                }
                .tag(1)
        }
    }
}

struct PhotoDetectionView: View {
    @ObservedObject var detector: ObjectDetector
    @Binding var threshold: Float
    let availableModels: [String]
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var detections: [Detection] = []
    @State private var inferenceTime: Double = 0
    @State private var isProcessing = false

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .overlay {
                            DetectionOverlayView(
                                detections: filteredDetections,
                                imageSize: image.size
                            )
                        }
                        .ignoresSafeArea()
                } else {
                    VStack(spacing: 16) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("Select a photo to detect objects")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(.systemGroupedBackground))
                }

                if isProcessing {
                    ProgressView("Detecting...")
                        .padding()
                        .background(.ultraThinMaterial)
                        .cornerRadius(10)
                }

                // Controls overlay
                VStack(spacing: 8) {
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
                        if selectedImage != nil {
                            Text("\(filteredDetections.count) detections")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(String(format: "%.0f ms", inferenceTime))
                                .font(.caption)
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
                ToolbarItem(placement: .topBarLeading) {
                    if availableModels.count > 1 {
                        Picker("Model", selection: Binding(
                            get: { detector.currentModelName },
                            set: { detector.switchModel($0) }
                        )) {
                            ForEach(availableModels, id: \.self) { name in
                                Text(name).tag(name)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Image(systemName: "photo.badge.plus")
                    }
                }
            }
            .onChange(of: selectedItem) { _, newItem in
                guard let newItem else { return }
                loadAndDetect(item: newItem)
            }
        }
    }

    private var filteredDetections: [Detection] {
        detections.filter { $0.confidence >= threshold }
    }

    private func loadAndDetect(item: PhotosPickerItem) {
        isProcessing = true
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data)
            else {
                isProcessing = false
                return
            }

            selectedImage = image
            let result = await detector.detect(image: image, threshold: 0.01)
            detections = result.detections
            inferenceTime = result.inferenceTime
            isProcessing = false
        }
    }
}
