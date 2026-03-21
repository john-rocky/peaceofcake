import SwiftUI

struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    @ObservedObject var detector: ObjectDetector
    @Binding var threshold: Float

    var body: some View {
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
                        .ignoresSafeArea()

                    GeometryReader { geometry in
                        CameraDetectionOverlayView(
                            detections: filteredDetections,
                            viewSize: geometry.size
                        )
                    }
                }
            }
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
    }

    private var filteredDetections: [Detection] {
        cameraManager.detections.filter { $0.confidence >= threshold }
    }
}

struct CameraDetectionOverlayView: View {
    let detections: [Detection]
    let viewSize: CGSize

    private let colors: [Color] = [
        .red, .green, .blue, .orange, .purple,
        .pink, .yellow, .cyan, .mint, .indigo,
    ]

    var body: some View {
        ForEach(detections) { detection in
            let rect = CGRect(
                x: detection.boundingBox.origin.x * viewSize.width,
                y: detection.boundingBox.origin.y * viewSize.height,
                width: detection.boundingBox.width * viewSize.width,
                height: detection.boundingBox.height * viewSize.height
            )
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
}
