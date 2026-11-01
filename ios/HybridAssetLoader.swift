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
        return try HybridAssetLoader.mapFile(atPath: url.path)
      }
      let data = try Data(contentsOf: url)
      return try ArrayBuffer.copy(data: data)
    }
  }

  /**
   * Memory-maps a local model instead of copying it into memory twice
   * (once into `Data`, once into the `ArrayBuffer`). `MAP_PRIVATE` is copy-on-write,
   * so the file on disk is never modified. The mapping is released when the
   * `ArrayBuffer` is deleted.
   */
  private static func mapFile(atPath path: String) throws -> ArrayBuffer {
    let fileDescriptor = open(path, O_RDONLY)
    guard fileDescriptor >= 0 else {
      throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                    userInfo: [NSLocalizedDescriptionKey: "Could not open \(path)"])
    }
    // The mapping stays valid after the descriptor is closed.
    defer { close(fileDescriptor) }

    var fileStat = stat()
    guard fstat(fileDescriptor, &fileStat) == 0 else {
      throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                    userInfo: [NSLocalizedDescriptionKey: "Could not stat \(path)"])
    }
    let size = Int(fileStat.st_size)
    guard size > 0 else {
      throw NSError(domain: "AssetLoader", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "TFLite model file is empty: \(path)"])
    }

    let mapped = mmap(nil, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileDescriptor, 0)
    guard let mapped, mapped != UnsafeMutableRawPointer(bitPattern: -1) else {
      throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                    userInfo: [NSLocalizedDescriptionKey: "Could not map \(path)"])
    }
    return ArrayBuffer.wrap(dataWithoutCopy: mapped, size: size) {
      munmap(mapped, size)
    }
  }
}
