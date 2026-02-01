import SwiftUI
import UIKit

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: [url],
            applicationActivities: nil
        )

        // Exclude activities that don't make sense for file sharing
        controller.excludedActivityTypes = [
            .addToReadingList,
            .assignToContact,
            .openInIBooks,
            .postToFacebook,
            .postToTwitter,
            .postToVimeo,
            .postToWeibo,
            .postToFlickr,
            .postToTencentWeibo
        ]

        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
