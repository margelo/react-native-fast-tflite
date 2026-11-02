package com.margelo.nitro.tflite

import android.annotation.SuppressLint
import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.NitroModules
import com.margelo.nitro.core.ArrayBuffer
import com.margelo.nitro.core.Promise
import java.net.URL

@Keep
@DoNotStrip
class HybridAssetLoader : HybridAssetLoaderSpec() {
  override fun loadAsset(path: String): Promise<ArrayBuffer> {
    return Promise.async {
      Log.i(TAG, "Loading TFLite model from \"$path\"...")
      // In release builds `require(..)` resolves to a scheme-less `res/raw` resource name,
      // which `java.net.URL` cannot parse. Load those from resources; everything with a
      // scheme (`http(s)://` dev server, `file://` local file) goes through URL.
      val bytes =
        when {
          path.startsWith("asset:/") -> loadFromAsset(path)
          path.contains("://") -> loadFromUrl(path)
          else -> loadFromRawResource(path)
        }
      Log.i(TAG, "Successfully loaded TFLite model from \"$path\"! (${bytes.size} bytes)")
      return@async ArrayBuffer.copy(bytes)
    }
  }

  private fun loadFromUrl(path: String): ByteArray {
    Log.i(TAG, "Loading \"$path\" from URL...")
    return URL(path).readBytes()
  }

  private fun loadFromAsset(path: String): ByteArray {
    Log.i(TAG, "Loading \"$path\" from assets...")
    val context =
      NitroModules.applicationContext
        ?: throw Error("Cannot load TFLite model - No Android Context available!")
    // `asset:/name` and `asset:///name` both refer to `name` in the APK's assets folder.
    return context.assets.open(path.removePrefix("asset:/").trimStart('/')).use { it.readBytes() }
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
