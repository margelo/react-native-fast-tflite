import Foundation
import NitroModules

class HybridAssetLoader: HybridAssetLoaderSpec {
  func loadAsset(path: String) throws -> Promise<ArrayBuffer> {
    return Promise.async {
      guard let url = path.hasPrefix("asset:/") ? bundleURL(forAsset: path) : URL(string: path) else {
        throw NSError(domain: "AssetLoader", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(path)"])
      }
      let data = try Data(contentsOf: url)
      return try ArrayBuffer.copy(data: data)
    }
  }
}

// `asset:/name` and `asset:///name` both refer to `name` in the app bundle.
private func bundleURL(forAsset path: String) -> URL? {
  let name = path.dropFirst("asset:/".count).drop(while: { $0 == "/" })
  return Bundle.main.resourceURL?.appendingPathComponent(String(name))
}
