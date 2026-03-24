import CoreImage
import CoreML
import UIKit
import Vision

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let labelIndex: Int
    let confidence: Float
    let boundingBox: CGRect // normalized 0-1 in image coordinates
}

class ObjectDetector: ObservableObject {
    private var model: MLModel?
    private let inputSize: CGFloat = 640
    private let ciContext = CIContext()

    init() {
        loadModel()
    }

    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "dfine_n_coco", withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: "dfine_n_coco", withExtension: "mlpackage")
        else {
            print("[D-FINE] Failed to find model in bundle")
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try MLModel(contentsOf: modelURL, configuration: config)
            print("[D-FINE] Model loaded successfully")
        } catch {
            print("[D-FINE] Failed to load model: \(error)")
        }
    }

    /// Async detection from UIImage (for photo picker)
    func detect(image: UIImage, threshold: Float) async -> (detections: [Detection], inferenceTime: Double) {
        guard let model = model else { return ([], 0) }
        guard let pixelBuffer = image.toPixelBuffer(size: CGSize(width: inputSize, height: inputSize)) else {
            return ([], 0)
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            let inputFeature = try MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)]
            )
            let output = try await model.prediction(from: inputFeature)
            let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            let detections = parseOutput(output, threshold: threshold)
            return (detections, inferenceTime)
        } catch {
            print("[D-FINE] Inference failed: \(error)")
            return ([], 0)
        }
    }

    /// Synchronous detection from CVPixelBuffer (for camera frames)
    /// Call from a background queue — not main thread.
    func detect(pixelBuffer: CVPixelBuffer, threshold: Float) -> (detections: [Detection], inferenceTime: Double) {
        guard let model = model else { return ([], 0) }
        guard let resizedBuffer = resizePixelBuffer(pixelBuffer, to: CGSize(width: inputSize, height: inputSize)) else {
            return ([], 0)
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            let inputFeature = try MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: resizedBuffer)]
            )
            let output = try model.prediction(from: inputFeature)
            let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            let detections = parseOutput(output, threshold: threshold)
            return (detections, inferenceTime)
        } catch {
            print("[D-FINE] Inference failed: \(error)")
            return ([], 0)
        }
    }

    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = size.width / ciImage.extent.width
        let scaleY = size.height / ciImage.extent.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var outputBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height),
                            kCVPixelFormatType_32BGRA, nil, &outputBuffer)
        guard let buffer = outputBuffer else { return nil }
        ciContext.render(scaled, to: buffer)
        return buffer
    }

    private func parseOutput(_ output: MLFeatureProvider, threshold: Float) -> [Detection] {
        guard let confidenceArray = output.featureValue(for: "confidence")?.multiArrayValue,
              let coordsArray = output.featureValue(for: "coordinates")?.multiArrayValue
        else { return [] }

        let numQueries = confidenceArray.shape[0].intValue
        let numClasses = confidenceArray.shape[1].intValue
        var detections: [Detection] = []

        for i in 0..<numQueries {
            // Find best class for this query
            var bestScore: Float = 0
            var bestClass: Int = 0
            for c in 0..<numClasses {
                let score = confidenceArray[[i, c] as [NSNumber]].floatValue
                if score > bestScore {
                    bestScore = score
                    bestClass = c
                }
            }
            guard bestScore >= threshold else { continue }

            let labelName = (bestClass >= 0 && bestClass < cocoLabels.count) ? cocoLabels[bestClass] : "class_\(bestClass)"

            // Coordinates are normalized cxcywh [0,1]
            let cx = CGFloat(coordsArray[[i, 0] as [NSNumber]].floatValue)
            let cy = CGFloat(coordsArray[[i, 1] as [NSNumber]].floatValue)
            let w  = CGFloat(coordsArray[[i, 2] as [NSNumber]].floatValue)
            let h  = CGFloat(coordsArray[[i, 3] as [NSNumber]].floatValue)

            let rect = CGRect(
                x: cx - w / 2,
                y: cy - h / 2,
                width: w,
                height: h
            )

            detections.append(Detection(
                label: labelName,
                labelIndex: bestClass,
                confidence: bestScore,
                boundingBox: rect
            ))
        }

        detections.sort { $0.confidence > $1.confidence }
        return detections
    }
}

extension UIImage {
    /// Normalize orientation and resize, then convert to CVPixelBuffer.
    func toPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        // Redraw with UIGraphicsImageRenderer to apply EXIF orientation
        let renderer = UIGraphicsImageRenderer(size: size)
        let normalized = renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }

        guard let cgImage = normalized.cgImage else { return nil }

        let width = Int(size.width)
        let height = Int(size.height)

        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        return buffer
    }
}
