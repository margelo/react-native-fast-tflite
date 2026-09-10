/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tflite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define TFLITE_XNNPACK_DELEGATE_FLAG_QS8 0x00000001
#define TFLITE_XNNPACK_DELEGATE_FLAG_QU8 0x00000002
#define TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16 0x00000004
#define TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED 0x00000008
#define TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS 0x00000010
#define TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER 0x00000020
#define TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS 0x00000040
#define TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING 0x00000080
#define TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SLINKY 0x00000100
#define TFLITE_XNNPACK_DELEGATE_FLAG_SLOW_CONSISTENT_ARITHMETIC 0x00000200
#define TFLITE_XNNPACK_DELEGATE_FLAG_DISABLE_SUBGRAPH_RESHAPING 0x00000400

struct TfLiteXNNPackDelegateWeightsCache;

typedef struct {
  int32_t num_threads;
  uint32_t runtime_flags;
  uint32_t flags;
  struct TfLiteXNNPackDelegateWeightsCache* weights_cache;
  bool handle_variable_ops;
  const char* weight_cache_file_path;
  int weight_cache_file_descriptor;
  void* weight_cache_provider;
} TfLiteXNNPackDelegateOptions;

TFL_CAPI_EXPORT bool TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider();

TFL_CAPI_EXPORT const char* TfLiteXNNPackDelegateInMemoryFilePath();

TFL_CAPI_EXPORT TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault();

TFL_CAPI_EXPORT TfLiteDelegate*
TfLiteXNNPackDelegateCreate(const TfLiteXNNPackDelegateOptions* options);

TfLiteDelegate*
TfLiteXNNPackDelegateCreateWithThreadpool(const TfLiteXNNPackDelegateOptions* options,
                                          TfLiteContext* context);

TFL_CAPI_EXPORT void* TfLiteXNNPackDelegateGetThreadPool(TfLiteDelegate* delegate);

TFL_CAPI_EXPORT const TfLiteXNNPackDelegateOptions*
TfLiteXNNPackDelegateGetOptions(TfLiteDelegate* delegate);

TFL_CAPI_EXPORT int TfLiteXNNPackDelegateGetFlags(TfLiteDelegate* delegate);

TFL_CAPI_EXPORT void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);

TFL_CAPI_EXPORT struct TfLiteXNNPackDelegateWeightsCache* TfLiteXNNPackDelegateWeightsCacheCreate();

TFL_CAPI_EXPORT struct TfLiteXNNPackDelegateWeightsCache*
TfLiteXNNPackDelegateWeightsCacheCreateWithSize(size_t size);

TFL_CAPI_EXPORT bool
TfLiteXNNPackDelegateWeightsCacheFinalizeSoft(struct TfLiteXNNPackDelegateWeightsCache* cache);

TFL_CAPI_EXPORT bool
TfLiteXNNPackDelegateWeightsCacheFinalizeHard(struct TfLiteXNNPackDelegateWeightsCache* cache);

TFL_CAPI_EXPORT void
TfLiteXNNPackDelegateWeightsCacheDelete(struct TfLiteXNNPackDelegateWeightsCache* cache);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_
