import Foundation
import NitroModules

class HybridAssetLoader: HybridAssetLoaderSpec {
  func loadAsset(path: String) throws -> Promise<ArrayBuffer> {
    return Promise.async {
      guard let url = URL(string: path) else {
        throw NSError(domain: "AssetLoader", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(path)"])
      }
      if url.isFileURL {
        return try HybridAssetLoader.mapFile(at: url)
      }
      let data = try Data(contentsOf: url)
      return try ArrayBuffer.copy(data: data)
    }
  }

  /**
   * Memory-maps a local model instead of copying it into memory twice
   * (once into `Data`, once into the `ArrayBuffer`).
   */
  private static func mapFile(at url: URL) throws -> ArrayBuffer {
    let data = try NSData(contentsOf: url, options: .mappedIfSafe)
    let bytes = UnsafeMutableRawPointer(mutating: data.bytes)

    return ArrayBuffer.wrap(dataWithoutCopy: bytes, size: data.length, onDelete: {
      // Keeps the file mapped until the ArrayBuffer is deleted.
      withExtendedLifetime(data) {}
    })
  }
}
