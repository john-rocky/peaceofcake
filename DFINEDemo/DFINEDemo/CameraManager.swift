import AVFoundation
import SwiftUI

class CameraManager: NSObject, ObservableObject {
    @Published var detections: [Detection] = []
    @Published var inferenceTime: Double = 0
    @Published var smoothedInferenceTime: Double = 0
    @Published var smoothedFPS: Double = 0
    @Published var isRunning = false
    @Published var permissionDenied = false
    @Published var frameSize = CGSize(width: 720, height: 1280)

    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "com.dfine.camera", qos: .userInitiated)

    private var detector: ObjectDetector?
    private var threshold: Float = 0.5
    private var isProcessingFrame = false
    private var frameSizeSet = false

    // Smoothing state
    private let smoothingAlpha: Double = 0.1
    private var frameTimestamps: [CFAbsoluteTime] = []
    private let maxTimestampCount = 30

    func setupCamera() {
        captureSession.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device)
        else { return }

        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }

        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90
        }
    }

    func checkPermissionAndStart(detector: ObjectDetector, threshold: Float) {
        self.detector = detector
        self.threshold = threshold

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.startSession()
                    } else {
                        self?.permissionDenied = true
                    }
                }
            }
        default:
            DispatchQueue.main.async { self.permissionDenied = true }
        }
    }

    private func startSession() {
        setupCamera()
        processingQueue.async { [weak self] in
            self?.captureSession.startRunning()
            DispatchQueue.main.async { self?.isRunning = true }
        }
    }

    func stop() {
        processingQueue.async { [weak self] in
            self?.captureSession.stopRunning()
            DispatchQueue.main.async { self?.isRunning = false }
        }
    }

    func updateThreshold(_ newThreshold: Float) {
        threshold = newThreshold
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !isProcessingFrame else { return }
        isProcessingFrame = true

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let detector = self.detector
        else {
            isProcessingFrame = false
            return
        }

        if !frameSizeSet {
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)
            frameSizeSet = true
            DispatchQueue.main.async { [weak self] in
                self?.frameSize = CGSize(width: w, height: h)
            }
        }

        let currentThreshold = threshold
        let result = detector.detect(pixelBuffer: pixelBuffer, threshold: currentThreshold)

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
        let alpha = smoothingAlpha
        if smoothedInferenceTime == 0 {
            emaInference = result.inferenceTime
            emaFPS = fps
        } else {
            emaInference = alpha * result.inferenceTime + (1 - alpha) * smoothedInferenceTime
            emaFPS = alpha * fps + (1 - alpha) * smoothedFPS
        }

        DispatchQueue.main.async { [weak self] in
            self?.detections = result.detections
            self?.inferenceTime = result.inferenceTime
            self?.smoothedInferenceTime = emaInference
            self?.smoothedFPS = emaFPS
            self?.isProcessingFrame = false
        }
    }
}
