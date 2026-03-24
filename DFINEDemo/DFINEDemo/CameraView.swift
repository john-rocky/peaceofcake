import SwiftUI

struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    @ObservedObject var detector: ObjectDetector
    @Binding var threshold: Float
    let availableModels: [String]

    var body: some View {
        NavigationStack {
        VStack(spacing: 0) {
            ZStack {
                if cameraManager.permissionDenied {
                    VStack(spacing: 16) {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("Camera access denied")
                            .foregroundColor(.secondary)
                        Text("Enable in Settings > Privacy > Camera")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(.systemGroupedBackground))
                } else {
                    CameraPreviewView(session: cameraManager.captureSession)

                    GeometryReader { geometry in
                        CameraDetectionOverlayView(
                            detections: filteredDetections,
                            viewSize: geometry.size,
                            frameSize: cameraManager.frameSize
                        )
                    }
                }
            }
            .clipped()
            .frame(maxWidth: .infinity, maxHeight: .infinity)

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
                    Text("\(filteredDetections.count) detections")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.0f ms", cameraManager.inferenceTime))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(.bar)
        }
        .onAppear {
            cameraManager.checkPermissionAndStart(detector: detector, threshold: threshold)
        }
        .onDisappear {
            cameraManager.stop()
        }
        .onChange(of: threshold) { _, newValue in
            cameraManager.updateThreshold(newValue)
        }
        .navigationTitle("D-FINE Camera")
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
        }
        } // NavigationStack
    }

    private var filteredDetections: [Detection] {
        cameraManager.detections.filter { $0.confidence >= threshold }
    }
}

struct CameraDetectionOverlayView: View {
    let detections: [Detection]
    let viewSize: CGSize
    let frameSize: CGSize

    private let colors: [Color] = [
        .red, .green, .blue, .orange, .purple,
        .pink, .yellow, .cyan, .mint, .indigo,
    ]

    var body: some View {
        let transform = aspectFillTransform()
        ForEach(detections) { detection in
            let rect = scaledRect(detection.boundingBox, transform: transform)
            let color = colors[detection.labelIndex % colors.count]

            Rectangle()
                .stroke(color, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)

            Text("\(detection.label) \(String(format: "%.0f%%", detection.confidence * 100))")
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(color.opacity(0.8))
                .cornerRadius(3)
                .position(x: rect.minX + 40, y: max(rect.minY - 10, 8))
        }
    }

    private struct FillTransform {
        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat
    }

    /// Compute how AVCaptureVideoPreviewLayer with .resizeAspectFill
    /// maps the camera frame onto the view.
    private func aspectFillTransform() -> FillTransform {
        guard frameSize.width > 0 && frameSize.height > 0 else {
            return FillTransform(scale: 1, offsetX: 0, offsetY: 0)
        }
        let scale = max(viewSize.width / frameSize.width,
                        viewSize.height / frameSize.height)
        let displayedWidth = frameSize.width * scale
        let displayedHeight = frameSize.height * scale
        let offsetX = (viewSize.width - displayedWidth) / 2
        let offsetY = (viewSize.height - displayedHeight) / 2
        return FillTransform(scale: scale, offsetX: offsetX, offsetY: offsetY)
    }

    /// Map a normalized bounding box (0-1, relative to full camera frame)
    /// to the view coordinates, accounting for aspectFill crop.
    private func scaledRect(_ normalizedRect: CGRect, transform t: FillTransform) -> CGRect {
        let displayedWidth = frameSize.width * t.scale
        let displayedHeight = frameSize.height * t.scale
        return CGRect(
            x: t.offsetX + normalizedRect.origin.x * displayedWidth,
            y: t.offsetY + normalizedRect.origin.y * displayedHeight,
            width: normalizedRect.width * displayedWidth,
            height: normalizedRect.height * displayedHeight
        )
    }
}
