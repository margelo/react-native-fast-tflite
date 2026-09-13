import { useEffect, useState } from 'react'
import type { TfliteModel, TensorflowModelDelegate } from './specs/Tflite.nitro'
import { loadTensorflowModel, type ModelSource } from './loadTensorflowModel'

export type TensorflowPlugin =
  | {
      model: TfliteModel
      state: 'loaded'
    }
  | {
      model: undefined
      state: 'loading'
    }
  | {
      model: undefined
      error: Error
      state: 'error'
    }

/**
 * Load a Tensorflow Lite Model from the given `.tflite` asset into a React State.
 *
 * * If you are passing in a `.tflite` model from your app's bundle using `require(..)`, make sure to add `tflite` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, make sure the URL points directly to a `.tflite` model. This can either be a web URL (`http://..`/`https://..`), or a local file (`file://..`).
 *
 * @param source The `.tflite` model in form of either a `require(..)` statement or a `{ url: string }`.
 * @param delegates The delegates to use for computations. Pass an empty array (`[]`) to use the standard CPU delegate. The `core-ml` (iOS), `android-gpu` and `nnapi` (Android) delegates are hardware-accelerated, but don't work on every model.
 * @returns The state of the Model.
 */
export function useTensorflowModel(
  source: ModelSource,
  delegates: TensorflowModelDelegate[]
): TensorflowPlugin {
  const [state, setState] = useState<TensorflowPlugin>({
    model: undefined,
    state: 'loading',
  })

  useEffect(() => {
    const load = async (): Promise<void> => {
      try {
        setState({ model: undefined, state: 'loading' })
        const m = await loadTensorflowModel(source, delegates)
        setState({ model: m, state: 'loaded' })
        console.log('Model loaded!')
      } catch (e) {
        console.error(`Failed to load Tensorflow Model ${source}!`, e)
        setState({ model: undefined, state: 'error', error: e as Error })
      }
    }
    load()
    // JSON.stringify compares delegates by value so inline array literals
    // (e.g. ['core-ml']) don't cause the effect to re-run every render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, JSON.stringify(delegates)])

  return state
}
