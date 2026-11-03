package com.margelo.nitro.tflite

import android.annotation.SuppressLint
import android.net.Uri
import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.NitroModules
import com.margelo.nitro.core.ArrayBuffer
import com.margelo.nitro.core.Promise
import java.io.File
import java.io.RandomAccessFile
import java.net.URL
import java.nio.channels.FileChannel

@Keep
@DoNotStrip
class HybridAssetLoader : HybridAssetLoaderSpec() {
  override fun loadAsset(path: String): Promise<ArrayBuffer> {
    return Promise.async {
      Log.i(TAG, "Loading TFLite model from \"$path\"...")
      if (path.startsWith("file://")) {
        return@async mapFile(File(Uri.decode(path.removePrefix("file://"))))
      }
      // In release builds `require(..)` resolves to a scheme-less `res/raw` resource name,
      // which `java.net.URL` cannot parse. Load those from resources; everything with a
      // scheme (`http(s)://` dev server) goes through URL. `file://` paths are mapped above.
      val bytes = if (path.contains("://")) loadFromUrl(path) else loadFromRawResource(path)
      Log.i(TAG, "Successfully loaded TFLite model from \"$path\"! (${bytes.size} bytes)")
      return@async ArrayBuffer.copy(bytes)
    }
  }

  private fun loadFromUrl(path: String): ByteArray {
    Log.i(TAG, "Loading \"$path\" from URL...")
    return URL(path).readBytes()
  }

  // Memory-maps a local model instead of reading it into the Java heap. Reading copies the
  // whole file into a ByteArray (and ByteArrayOutputStream grows to about twice its size),
  // so a model close to the app's heap limit (often 256 MB) throws OutOfMemoryError.
  // The mapping is read-only (TFLite only reads the model) and stays valid after the
  // channel is closed; the wrapped direct ByteBuffer keeps it alive.
  private fun mapFile(file: File): ArrayBuffer {
    Log.i(TAG, "Memory-mapping \"${file.path}\"...")
    RandomAccessFile(file, "r").use { randomAccessFile ->
      val size = randomAccessFile.length()
      if (size == 0L) {
        throw Error("Cannot load TFLite model - \"${file.path}\" is empty!")
      }
      val mapped = randomAccessFile.channel.map(FileChannel.MapMode.READ_ONLY, 0, size)
      Log.i(TAG, "Successfully mapped TFLite model from \"${file.path}\"! ($size bytes)")
      return ArrayBuffer.wrap(mapped)
    }
  }

  @SuppressLint("DiscouragedApi")
  private fun loadFromRawResource(path: String): ByteArray {
    Log.i(TAG, "Loading \"$path\" from res/raw...")
    val context =
      NitroModules.applicationContext
        ?: throw Error("Cannot load TFLite model - No Android Context available!")
    val rawResourceId: Int = context.resources.getIdentifier(path, "raw", context.packageName)
    if (rawResourceId == 0) {
      throw Error("Cannot find TFLite model \"$path\" in res/raw!")
    }
    Log.i(TAG, "Resolved \"$path\" to res/raw resource ID $rawResourceId.")
    return context.resources.openRawResource(rawResourceId).use { it.readBytes() }
  }
}
