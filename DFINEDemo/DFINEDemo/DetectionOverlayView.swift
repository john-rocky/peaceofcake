import SwiftUI

struct DetectionOverlayView: View {
    let detections: [Detection]
    let imageSize: CGSize

    private let colors: [Color] = [
        .red, .green, .blue, .orange, .purple,
        .pink, .yellow, .cyan, .mint, .indigo,
    ]

    var body: some View {
        GeometryReader { geometry in
            let displaySize = geometry.size
            ForEach(detections) { detection in
                let rect = scaledRect(detection.boundingBox, in: displaySize)
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

    private func scaledRect(_ normalizedRect: CGRect, in displaySize: CGSize) -> CGRect {
        // The image is displayed with aspect fit, so we need to compute the
        // actual displayed image rect within the view.
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = displaySize.width / displaySize.height

        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat

        if imageAspect > viewAspect {
            // Image wider than view — pillarboxed (black bars top/bottom)
            scale = displaySize.width / imageSize.width
            offsetX = 0
            offsetY = (displaySize.height - imageSize.height * scale) / 2
        } else {
            // Image taller than view — letterboxed (black bars left/right)
            scale = displaySize.height / imageSize.height
            offsetX = (displaySize.width - imageSize.width * scale) / 2
            offsetY = 0
        }

        let scaledImageWidth = imageSize.width * scale
        let scaledImageHeight = imageSize.height * scale

        return CGRect(
            x: offsetX + normalizedRect.origin.x * scaledImageWidth,
            y: offsetY + normalizedRect.origin.y * scaledImageHeight,
            width: normalizedRect.width * scaledImageWidth,
            height: normalizedRect.height * scaledImageHeight
        )
    }
}
