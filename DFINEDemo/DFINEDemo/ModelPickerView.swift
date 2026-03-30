import SwiftUI

/// Compact model picker shown in the controls overlay.
struct ModelPickerView: View {
    @ObservedObject var detector: ObjectDetector
    let availableModels: [String]

    var body: some View {
        if availableModels.count > 1 {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(availableModels, id: \.self) { name in
                        ModelChip(
                            label: displayName(name),
                            isSelected: detector.currentModelName == name
                        ) {
                            detector.switchModel(name)
                        }
                    }
                }
            }
        }
    }

    private func displayName(_ name: String) -> String {
        // "dfine_n_coco" -> "D-FINE N", "rfdetr_n_coco" -> "RF-DETR N"
        let lower = name.lowercased()
        let family: String
        let rest: String
        if lower.hasPrefix("dfine") {
            family = "D-FINE"
            rest = String(lower.dropFirst(5))
        } else if lower.hasPrefix("rfdetr") {
            family = "RF-DETR"
            rest = String(lower.dropFirst(6))
        } else {
            return name
        }

        let sizeMap = ["_n_": " N", "_s_": " S", "_m_": " M", "_l_": " L", "_x_": " X",
                       "-n-": " N", "-s-": " S", "-m-": " M", "-l-": " L", "-x-": " X"]
        for (pattern, label) in sizeMap {
            if rest.contains(pattern) { return family + label }
        }
        return family
    }
}

private struct ModelChip: View {
    let label: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.caption)
                .fontWeight(isSelected ? .bold : .regular)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(isSelected ? Color.accentColor : Color.secondary.opacity(0.2))
                .foregroundColor(isSelected ? .white : .primary)
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}
