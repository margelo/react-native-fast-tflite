import { Image } from 'react-native'
import { NitroModules } from 'react-native-nitro-modules'
import type {
  AssetLoader,
  TfliteModel,
  TfliteModule,
  TensorflowModelDelegate,
} from './specs/Tflite.nitro'

// In React Native, `require(..)` returns a number.
type Require = number
export type ModelSource = Require | { url: string }

const assetLoader = NitroModules.createHybridObject<AssetLoader>('AssetLoader')
const tfliteModule =
  NitroModules.createHybridObject<TfliteModule>('TfliteModule')

/**
 * Load a Tensorflow Lite Model from the given `.tflite` asset.
 *
 * * If you are passing in a `.tflite` model from your app's bundle using `require(..)`, make sure to add `tflite` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, make sure the URL points directly to a `.tflite` model. This can either be a web URL (`http://..`/`https://..`), or a local file (`file://..`).
 *
 * @param source The `.tflite` model in form of either a `require(..)` statement or a `{ url: string }`.
 * @param delegates The delegates to use for computations. Pass an empty array (`[]`) to use the standard CPU delegate. The `core-ml` (iOS), `android-gpu` and `nnapi` (Android) delegates are hardware-accelerated, but don't work on every model.
 * @returns The loaded Model.
 */
export async function loadTensorflowModel(
  source: ModelSource,
  delegates: TensorflowModelDelegate[]
): Promise<TfliteModel> {
  let uri: string
  if (typeof source === 'number') {
    console.log(`Loading Tensorflow Lite Model ${source}`)
    const asset = Image.resolveAssetSource(source)
    uri = asset.uri
    console.log(`Resolved Model path: ${asset.uri}`)
  } else if (typeof source === 'object' && 'url' in source) {
    uri = source.url
  } else {
    throw new Error(
      'TFLite: Invalid source passed! Source should be either a React Native require(..) or a `{ url: string }` object!'
    )
  }
  const data = await assetLoader.loadAsset(uri)
  return tfliteModule.createModel(data, delegates)
}
