import type { HybridObject } from 'react-native-nitro-modules'

/**
 * A hardware-accelerating delegate to run the TFLite model with.
 *
 * - `'core-ml'`: CoreML (Apple platforms). Requires `$EnableCoreMLDelegate`.
 * - `'nnapi'`: NNAPI (Android)
 * - `'android-gpu'`: GPU (Android)
 * - `'metal'`: **deprecated, not implemented** - creating a model with this
 *   delegate always throws. Use `'core-ml'` on Apple platforms instead.
 */
export type TensorflowModelDelegate =
  | 'metal'
  | 'core-ml'
  | 'nnapi'
  | 'android-gpu'

export type TensorDataType =
  | 'string'
  | 'float16'
  | 'float32'
  | 'float64'
  | 'bfloat16'
  | 'int4'
  | 'int8'
  | 'int16'
  | 'int32'
  | 'int64'
  | 'uint8'
  | 'uint16'
  | 'uint32'
  | 'uint64'
  | 'bool'
  | 'complex64'
  | 'complex128'
  | 'resource'
  | 'variant'
  | 'none'

export interface Tensor {
  name: string
  dataType: TensorDataType
  shape: number[]
}

export interface TfliteModel
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  readonly delegates: TensorflowModelDelegate[]
  readonly inputs: Tensor[]
  readonly outputs: Tensor[]
  runSync(input: ArrayBuffer[]): ArrayBuffer[]
  run(input: ArrayBuffer[]): Promise<ArrayBuffer[]>
}

export interface TfliteModule
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  /**
   * Create a new {@linkcode TfliteModel} with the given
   * {@linkcode modelData} (a binary representation of the
   * TFLite model), and optionally a list of hardware
   * accelerating {@linkcode TensorflowModelDelegate}s.
   *
   * If {@linkcode delegates} is empty (`[]`), the default
   * CPU delegate will be used.
   */
  createModel(
    modelData: ArrayBuffer,
    delegates: TensorflowModelDelegate[]
  ): TfliteModel
}

export interface AssetLoader
  extends HybridObject<{ ios: 'swift'; android: 'kotlin' }> {
  /**
   * Load an asset from the given {@linkcode path} and
   * return its contents as an {@linkcode ArrayBuffer}.
   */
  loadAsset(path: string): Promise<ArrayBuffer>
}
