# Migrating from v2 to v3

`react-native-fast-tflite` v3 is a major rewrite that migrates from TurboModules to [Nitro Modules](https://github.com/mrousavy/nitro). The public API remains similar, but there are several breaking changes.

## 1. Install `react-native-nitro-modules`

v3 requires `react-native-nitro-modules` as a peer dependency:

```sh
# npm
npm install react-native-fast-tflite react-native-nitro-modules

# yarn
yarn add react-native-fast-tflite react-native-nitro-modules
```

Then rebuild native projects:

```sh
cd ios && pod install && cd ..
```

## 2. Delegates: string -> array


```diff
- const model = await loadTensorflowModel(source)
+ const model = await loadTensorflowModel(source, [])
```

The `delegate` parameter has been renamed to `delegates` and now accepts an **array** instead of a single string.

```diff
- const model = await loadTensorflowModel(source, 'core-ml')
+ const model = await loadTensorflowModel(source, ['core-ml'])
```

```diff
- const model = useTensorflowModel(source, 'core-ml')
+ const model = useTensorflowModel(source, ['core-ml'])
```

### The `'metal'` delegate is not implemented

`'metal'` is still part of the `TensorflowModelDelegate` union, but it has no
implementation - creating a model with it throws
`The Metal Delegate ("metal") is not implemented!`
(see [`cpp/TfliteHelpers.cpp`](./cpp/TfliteHelpers.cpp), `getMetalDelegate()`).
It is documented as deprecated on the `TensorflowModelDelegate` type. On Apple
platforms, use `'core-ml'` instead:

```diff
- const model = useTensorflowModel(source, ['metal'])
+ const model = useTensorflowModel(source, ['core-ml'])
```

### The `'default'` delegate is removed

To use CPU-only inference (previously `'default'`), pass an empty array:

```diff
- const model = await loadTensorflowModel(source, 'default')
+ const model = await loadTensorflowModel(source, [])
```

### `model.delegate` -> `model.delegates`

```diff
- console.log(model.delegate)   // 'core-ml'
+ console.log(model.delegates)  // ['core-ml']
```

## 3. Inputs/Outputs: TypedArray -> ArrayBuffer

`run()` and `runSync()` now accept and return `ArrayBuffer[]` instead of `TypedArray[]`.

### Preparing inputs

```diff
- const output = model.runSync([float32Array])
+ const output = model.runSync([float32Array.buffer])
```

If your TypedArray is a view into a larger buffer, slice it first:

```ts
const inputBuffer = typedArray.buffer.slice(
  typedArray.byteOffset,
  typedArray.byteOffset + typedArray.byteLength
)
const output = model.runSync([inputBuffer])
```

### Reading outputs

```diff
  const outputs = model.runSync([input])
- const scores = outputs[0]  // was already a Float32Array
+ const scores = new Float32Array(outputs[0]!)  // wrap ArrayBuffer manually
```

Choose the appropriate TypedArray constructor based on your model's output tensor data type (e.g., `Uint8Array`, `Int32Array`, `Float32Array`).

## 4. VisionCamera v4 worklet compatibility

Because `TfliteModel` is now a Nitro `HybridObject`, VisionCamera v4's worklet runtime cannot access it directly. You need to box/unbox it:

```ts
import { NitroModules } from 'react-native-nitro-modules'

const model = await loadTensorflowModel(source, ['core-ml'])
const boxedModel = NitroModules.box(model)

// Inside a VisionCamera V4 frame processor worklet:
const unboxedModel = boxedModel.unbox()
const output = unboxedModel.runSync([inputBuffer])
```

> This workaround will not be needed with VisionCamera v5.

## 5. Android gradle.properties

If you had custom property overrides using the `Tflite_` prefix, rename them to `NitroTflite_`:

```diff
- Tflite_enableGpuDelegate=true
+ NitroTflite_enableGpuDelegate=true
```

## 6. Expanded `TensorDataType`

`Tensor.dataType` now includes additional types: `string`, `bfloat16`, `int4`, `uint16`, `uint32`, `uint64`, `complex64`, `complex128`, `resource`, `variant`, `none`.

The `'invalid'` type has been removed. If you had exhaustive switch/case handling over `dataType`, update it accordingly.

