<a href="https://margelo.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./img/banner-dark.webp" />
    <source media="(prefers-color-scheme: light)" srcset="./img/banner-light.webp" />
    <img alt="Fast TFLite" src="./img/banner-light.webp" />
  </picture>
</a>

A high-performance [TensorFlow Lite](https://www.tensorflow.org/lite) library for React Native, built with [Nitro Modules](https://github.com/mrousavy/nitro).

- ⚡ Powered by Nitro Modules
- 💨 Zero-copy ArrayBuffers
- 🔧 Uses the low-level C/C++ TensorFlow Lite core API for direct memory access
- 🔄 Supports swapping out TensorFlow Models at runtime
- 🖥️ Supports hardware-accelerated delegates (CoreML on iOS, GPU/NNAPI on Android)
- 📸 Easy [VisionCamera](https://github.com/mrousavy/react-native-vision-camera) integration

## Migrating from v2

If you are upgrading from v2, see the [Migration Guide](./MIGRATION_V2_TO_V3.md) for breaking changes and upgrade steps.

## Installation

1. Add the npm packages:
   ```sh
   yarn add react-native-fast-tflite react-native-nitro-modules
   ```
2. In `metro.config.js`, add `tflite` as a supported asset extension:
   ```js
   module.exports = {
     // ...
     resolver: {
       assetExts: ['tflite', // ...
       // ...
   ```
   This allows you to drop `.tflite` files into your app and swap them out at runtime without rebuilding. 🔥
3. (Optional) To enable GPU delegates, see [Using GPU Delegates](#using-gpu-delegates) below.
4. Run your app (`yarn android` / `npx pod-install && yarn ios`)

## Usage

1. Find a TensorFlow Lite (`.tflite`) model. There are thousands of public models on [tfhub.dev](https://tfhub.dev).
2. Drag your model into your app's asset folder (e.g. `src/assets/my-model.tflite`)
3. Load the Model:

   ```ts
   // Option A: Standalone Function
   const model = await loadTensorflowModel(require('assets/my-model.tflite'), [])

   // Option B: Hook in a Function Component
   const plugin = useTensorflowModel(require('assets/my-model.tflite'), [])
   ```

4. Call the Model:
   ```ts
   const inputData: ArrayBuffer = ...
   const outputData = await model.run([inputData])
   console.log(outputData)
   ```

### Loading Models

Models can be loaded from the React Native bundle via `require(..)`, or any URI/URL (`http://..` or `file://..`):

```ts
// Asset from React Native Bundle
loadTensorflowModel(require('assets/my-model.tflite'), [])
// File on the local filesystem
loadTensorflowModel({ url: 'file:///var/mobile/.../my-model.tflite' }, [])
// Remote URL
loadTensorflowModel(
  { url: 'https://tfhub.dev/google/lite-model/object_detection_v1.tflite' },
  []
)
```

Loading a Model is asynchronous since buffers need to be allocated. Make sure to handle errors when loading.

### Input and Output data

TensorFlow uses _tensors_ as input and output. Since TensorFlow Lite is optimized for fixed-size byte buffers, you are responsible for interpreting the raw data yourself.

Input and output values are passed as `ArrayBuffer`. To inspect tensor shapes, open your model in [Netron](https://netron.app).

For example, the `object_detection_mobile_object_localizer_v1_1_default_1.tflite` model on [tfhub.dev](https://tfhub.dev) has **1 input tensor** and **4 output tensors**:

![Screenshot of netron.app inspecting the model](./img/netron-inspect-model.png)

In the description on [tfhub.dev](https://tfhub.dev) we can find the description of all tensors:

![Screenshot of tfhub.dev inspecting the model](./img/tfhub-description.png)

From that we know we need a 192 x 192 input image with 3 bytes per pixel (RGB).

#### Usage (VisionCamera)

If you're using this model with a [VisionCamera](https://github.com/mrousavy/react-native-vision-camera) Frame Processor, you need to convert the Frame to the model's expected input size.
Use [vision-camera-resizer](https://visioncamera.margelo.com/api/react-native-vision-camera-resizer) to do the conversion:

```tsx
import { Camera, useFrameOutput } from 'react-native-vision-camera'
import { useResizer } from 'react-native-vision-camera-resizer'
import { useTensorflowModel } from 'react-native-fast-tflite'

const objectDetection = useTensorflowModel(require('object_detection.tflite'), [])

// 1. Create a resizer that converts Frames to 192x192x3 (RGB, uint8)
const { resizer } = useResizer({
  width: 192,
  height: 192,
  channelOrder: 'rgb',
  dataType: 'uint8',
})

const frameOutput = useFrameOutput({
  pixelFormat: 'yuv',
  onFrame(frame) {
    'worklet'
    if (objectDetection.state !== 'loaded' || resizer == null) {
      frame.dispose()
      return
    }

    // 2. Resize the Frame to the model's input size
    const resized = resizer.resize(frame)
    frame.dispose()
    const data = new Uint8Array(resized.getPixelBuffer())
    resized.dispose()

    // 3. Extract the exact slice of the underlying ArrayBuffer
    const inputBuffer = data.buffer.slice(
      data.byteOffset,
      data.byteOffset + data.byteLength
    )

    // 4. Run model with given input buffer synchronously
    const outputs = objectDetection.model.runSync([inputBuffer])

    // 5. Interpret outputs accordingly
    const detection_boxes = new Float32Array(outputs[0]!)
    const detection_classes = new Float32Array(outputs[1]!)
    const detection_scores = new Float32Array(outputs[2]!)
    const num_detections = new Float32Array(outputs[3]!)
    console.log(`Detected ${num_detections[0]} objects!`)
  },
})

return <Camera device="back" isActive={true} outputs={[frameOutput]} {...otherProps} />
```

> [!NOTE]
> Unlike v4, VisionCamera v5 no longer requires boxing the model with `NitroModules.box()`. Since v5 is built on Nitro Modules and uses [react-native-worklets](https://docs.swmansion.com/react-native-worklets/), worklets can access HybridObjects like the TFLite model directly.

### Using GPU Delegates

GPU Delegates offer faster, GPU-accelerated computation. There are multiple delegates available:

#### CoreML (iOS)

##### Expo

Use the config plugin in your expo config (`app.json`, `app.config.json` or `app.config.js`):

```json
{
  "name": "my app",
  "plugins": [
    [
      "react-native-fast-tflite",
      {
        "enableCoreMLDelegate": true
      }
    ]
  ]
}
```

##### Bare React Native

1. Set `$EnableCoreMLDelegate` to `true` in your `Podfile`:

   ```ruby
   $EnableCoreMLDelegate=true

   # rest of your podfile...
   ```

2. Open your iOS project in Xcode and add the `CoreML` framework under **General → Frameworks, Libraries and Embedded Content**.
3. Re-install Pods and build:
   ```sh
   cd ios && pod install && cd ..
   yarn ios
   ```
4. Use the CoreML Delegate:
   ```ts
   const model = await loadTensorflowModel(
     require('assets/my-model.tflite'),
     ['core-ml']
   )
   ```

> [!NOTE]
> Not all model operations are supported on the CoreML delegate. Make sure your model is compatible.

#### Android GPU/NNAPI (Android)

To enable GPU or NNAPI on Android, you **may** need to include native libraries, especially on Android 12+.

##### Expo

Use the config plugin in your expo config (`app.json`, `app.config.json` or `app.config.js`) with `enableAndroidGpuLibraries`:

```json
{
  "name": "my app",
  "plugins": [
    [
      "react-native-fast-tflite",
      {
        "enableAndroidGpuLibraries": true
      }
    ]
  ]
}
```

By default, when enabled, `libOpenCL.so` will be included in your `AndroidManifest.xml`. You can also include more libraries by passing an array:

```json
{
  "name": "my app",
  "plugins": [
    [
      "react-native-fast-tflite",
      {
        "enableAndroidGpuLibraries": ["libOpenCL-pixel.so", "libGLES_mali.so"]
      }
    ]
  ]
}
```

> [!NOTE]
> For Expo, remember to run prebuild if the library is not yet included in your `AndroidManifest.xml`.

##### Bare React Native

Add any needed entries to your `AndroidManifest.xml`:

```xml
<uses-native-library android:name="libOpenCL.so" android:required="false" />
<uses-native-library android:name="libOpenCL-pixel.so" android:required="false" />
<uses-native-library android:name="libGLES_mali.so" android:required="false" />
<uses-native-library android:name="libPVROCL.so" android:required="false" />
```

Then use the delegate:

```ts
const model = await loadTensorflowModel(
  require('assets/my-model.tflite'),
  ['android-gpu']
)
// or
const model = await loadTensorflowModel(
  require('assets/my-model.tflite'),
  ['nnapi']
)
```

> [!WARNING]
> NNAPI is deprecated on Android 15. GPU delegate is preferred.

> [!NOTE]
> Android does not officially support OpenCL, but most GPU vendors do.

## Community Discord

[Join the Margelo Community Discord](https://discord.gg/6CSHz2qAvA) to chat about react-native-fast-tflite or other Margelo libraries.

## Adopting at scale

<a href="https://github.com/sponsors/mrousavy">
  <img align="right" width="160" alt="This library helped you? Consider sponsoring!" src=".github/funding-octocat.svg">
</a>

This library is provided _as is_, I work on it in my free time.

If you're integrating react-native-fast-tflite in a production app, consider [funding this project](https://github.com/sponsors/mrousavy) and <a href="mailto:me@mrousavy.com?subject=Adopting react-native-fast-tflite at scale">contact me</a> to receive premium enterprise support, help with issues, prioritize bugfixes, request features, and more.

## Contributing

1. Clone the repo
2. Make sure you have installed Xcode CLI tools such as `gcc`, `cmake` and `python`/`python3`. See the TensorFlow documentation on what you need exactly.
3. Run `yarn bootstrap` and select `y` on all iOS and Android related questions.
4. Open the example app and start developing
   - iOS: `example/ios/TfliteExample.xcworkspace`
   - Android: `example/android`

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT
