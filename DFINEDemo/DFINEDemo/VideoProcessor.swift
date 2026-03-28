import AVFoundation
import CoreImage
import UIKit

class VideoProcessor: ObservableObject {
    @Published var detections: [Detection] = []
    @Published var currentFrame: UIImage?
    @Published var inferenceTime: Double = 0
    @Published var smoothedInferenceTime: Double = 0
    @Published var smoothedFPS: Double = 0
    @Published var isPlaying = false
    @Published var isProcessing = false
    @Published var progress: Double = 0
    @Published var videoSize: CGSize = .zero

    private var detector: ObjectDetector?
    private var threshold: Float = 0.5
    private var reader: AVAssetReader?
    private var playbackTask: Task<Void, Never>?
    private let ciContext = CIContext()

    private let smoothingAlpha: Double = 0.1
    private var frameTimestamps: [CFAbsoluteTime] = []
    private let maxTimestampCount = 30

    func setup(detector: ObjectDetector, threshold: Float) {
        self.detector = detector
        self.threshold = threshold
    }

    func updateThreshold(_ newThreshold: Float) {
        threshold = newThreshold
    }

    func play(url: URL) {
        stop()
        playbackTask = Task.detached { [weak self] in
            await self?.processVideo(url: url)
        }
    }

    func stop() {
        playbackTask?.cancel()
        playbackTask = nil
        reader?.cancelReading()
        reader = nil
        DispatchQueue.main.async { [weak self] in
            self?.isPlaying = false
            self?.smoothedInferenceTime = 0
            self?.smoothedFPS = 0
            self?.frameTimestamps.removeAll()
        }
    }

    private func processVideo(url: URL) async {
        let asset = AVURLAsset(url: url)

        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return }
        let duration = try? await asset.load(.duration)
        let totalSeconds = duration.map { CMTimeGetSeconds($0) } ?? 1
        let naturalSize = try? await track.load(.naturalSize)
        let transform = try? await track.load(.preferredTransform)
        let nominalFrameRate = try? await track.load(.nominalFrameRate)
        let frameInterval = 1.0 / Double(nominalFrameRate ?? 30)

        var resolvedVideoSize = CGSize.zero
        if let size = naturalSize, let t = transform {
            let transformed = CGRectApplyAffineTransform(
                CGRect(origin: .zero, size: size), t
            )
            resolvedVideoSize = CGSize(width: abs(transformed.width), height: abs(transformed.height))
        }

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        guard let assetReader = try? AVAssetReader(asset: asset) else { return }
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        trackOutput.alwaysCopiesSampleData = false
        assetReader.add(trackOutput)

        guard assetReader.startReading() else { return }
        self.reader = assetReader

        await MainActor.run {
            self.videoSize = resolvedVideoSize
            self.isPlaying = true
            self.isProcessing = true
        }

        while !Task.isCancelled && assetReader.status == .reading {
            guard let sampleBuffer = trackOutput.copyNextSampleBuffer() else { break }

            let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let currentSeconds = CMTimeGetSeconds(timestamp)
            let currentProgress = currentSeconds / totalSeconds

            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
                  let detector = self.detector else { continue }

            let currentThreshold = threshold
            let result = detector.detect(pixelBuffer: pixelBuffer, threshold: currentThreshold)
            let frameImage = pixelBufferToUIImage(pixelBuffer)

            let now = CFAbsoluteTimeGetCurrent()
            frameTimestamps.append(now)
            if frameTimestamps.count > maxTimestampCount {
                frameTimestamps.removeFirst(frameTimestamps.count - maxTimestampCount)
            }

            var fps: Double = 0
            if frameTimestamps.count >= 2,
               let first = frameTimestamps.first, let last = frameTimestamps.last, last > first {
                fps = Double(frameTimestamps.count - 1) / (last - first)
            }

            let emaInference: Double
            let emaFPS: Double
            if smoothedInferenceTime == 0 {
                emaInference = result.inferenceTime
                emaFPS = fps
            } else {
                emaInference = smoothingAlpha * result.inferenceTime + (1 - smoothingAlpha) * smoothedInferenceTime
                emaFPS = smoothingAlpha * fps + (1 - smoothingAlpha) * smoothedFPS
            }

            await MainActor.run {
                self.detections = result.detections
                self.inferenceTime = result.inferenceTime
                self.smoothedInferenceTime = emaInference
                self.smoothedFPS = emaFPS
                self.currentFrame = frameImage
                self.progress = currentProgress
            }

            // Pace playback to approximate real-time
            let sleepTime = max(frameInterval - result.inferenceTime / 1000, 0)
            if sleepTime > 0 {
                try? await Task.sleep(for: .seconds(sleepTime))
            }
        }

        await MainActor.run {
            self.isPlaying = false
            self.isProcessing = false
            self.progress = 1.0
        }
    }

    private func pixelBufferToUIImage(_ pixelBuffer: CVPixelBuffer) -> UIImage {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return UIImage()
        }
        return UIImage(cgImage: cgImage)
    }
}
