import type { HybridObject } from 'react-native-nitro-modules'

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
  /**
   * Same as {@linkcode createModel}, but builds the interpreter
   * on a background thread instead of blocking the calling one.
   *
   * Building the interpreter can take seconds: hardware
   * delegates such as `core-ml` compile the model while it is
   * created, which would otherwise freeze the JS thread.
   *
   * {@linkcode modelData} is read on the calling thread, and
   * copied there first unless native code owns it (as
   * `AssetLoader.loadAsset(...)`'s result does), since a JS-owned
   * buffer may only be read on the JS thread.
   */
  createModelAsync(
    modelData: ArrayBuffer,
    delegates: TensorflowModelDelegate[]
  ): Promise<TfliteModel>
}

export interface AssetLoader
  extends HybridObject<{ ios: 'swift'; android: 'kotlin' }> {
  /**
   * Load an asset from the given {@linkcode path} and
   * return its contents as an {@linkcode ArrayBuffer}.
   */
  loadAsset(path: string): Promise<ArrayBuffer>
}
