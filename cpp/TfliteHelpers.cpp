#include "TfliteHelpers.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <thread>

#if defined(ANDROID)
#include "vendor/litert/xnnpack_delegate.h"
#include <tflite/c/c_api.h>
#include <tflite/delegates/gpu/delegate.h>
#include <tflite/delegates/nnapi/nnapi_delegate_c_api.h>
#elif defined(__APPLE__)
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#if FAST_TFLITE_ENABLE_CORE_ML
#include <TensorFlowLiteCCoreML/TensorFlowLiteCCoreML.h>
#endif
#else
#error "Invalid Platform!"
#endif

namespace margelo::nitro::tflite {

namespace {

  constexpr uint32_t MIN_XNNPACK_THREADS = 1;
  constexpr uint32_t MAX_XNNPACK_THREADS = 4;

  int32_t getXNNPACKThreadCount() {
    const uint32_t detectedThreadCount = std::thread::hardware_concurrency();
    if (detectedThreadCount <= MIN_XNNPACK_THREADS) {
      return static_cast<int32_t>(MIN_XNNPACK_THREADS);
    }
    const uint32_t reservedThreadCount = detectedThreadCount - 1;
    return static_cast<int32_t>(
        std::clamp(reservedThreadCount, MIN_XNNPACK_THREADS, MAX_XNNPACK_THREADS));
  }

} // namespace

// TODO: Remove this, this doesn't seem like a good idea at all.
typedef float float32_t;
typedef double float64_t;

std::string tfLiteStatusToString(TfLiteStatus status) {
  switch (status) {
    case kTfLiteOk:
      return "ok";
    case kTfLiteError:
      return "error";
    case kTfLiteDelegateError:
      return "delegate-error";
    case kTfLiteApplicationError:
      return "application-error";
    case kTfLiteDelegateDataNotFound:
      return "delegate-data-not-found";
    case kTfLiteDelegateDataWriteError:
      return "delegate-data-write-error";
    case kTfLiteDelegateDataReadError:
      return "delegate-data-read-error";
    case kTfLiteUnresolvedOps:
      return "unresolved-ops";
    case kTfLiteCancelled:
      return "cancelled";
    default:
      return "unknown";
  }
}

TensorDataType getTensorDataType(TfLiteType dataType) {
  switch (dataType) {
    case kTfLiteFloat16:
      return TensorDataType::FLOAT16;
    case kTfLiteFloat32:
      return TensorDataType::FLOAT32;
    case kTfLiteFloat64:
      return TensorDataType::FLOAT64;
    case kTfLiteBFloat16:
      return TensorDataType::BFLOAT16;
    case kTfLiteInt4:
      return TensorDataType::INT4;
    case kTfLiteInt8:
      return TensorDataType::INT8;
    case kTfLiteInt16:
      return TensorDataType::INT16;
    case kTfLiteInt32:
      return TensorDataType::INT32;
    case kTfLiteInt64:
      return TensorDataType::INT64;
    case kTfLiteUInt8:
      return TensorDataType::UINT8;
    case kTfLiteUInt16:
      return TensorDataType::UINT16;
    case kTfLiteUInt32:
      return TensorDataType::UINT32;
    case kTfLiteUInt64:
      return TensorDataType::UINT64;
    case kTfLiteNoType:
      return TensorDataType::NONE;
    case kTfLiteString:
      return TensorDataType::STRING;
    case kTfLiteBool:
      return TensorDataType::BOOL;
    case kTfLiteComplex64:
      return TensorDataType::COMPLEX64;
    case kTfLiteComplex128:
      return TensorDataType::COMPLEX128;
    case kTfLiteResource:
      return TensorDataType::RESOURCE;
    case kTfLiteVariant:
      return TensorDataType::VARIANT;
  }
  throw std::runtime_error("Unsupported TfLiteType: " + std::to_string(static_cast<int>(dataType)));
}

size_t getTFLTensorDataTypeSize(TfLiteType dataType) {
  switch (dataType) {
    case kTfLiteBool:
      return sizeof(bool);
    case kTfLiteFloat32:
      return sizeof(float32_t);
    case kTfLiteInt32:
      return sizeof(int32_t);
    case kTfLiteUInt8:
      return sizeof(uint8_t);
    case kTfLiteInt64:
      return sizeof(int64_t);
    case kTfLiteInt16:
      return sizeof(int16_t);
    case kTfLiteInt8:
      return sizeof(int8_t);
    case kTfLiteFloat64:
      return sizeof(float64_t);
    case kTfLiteUInt64:
      return sizeof(uint64_t);
    case kTfLiteUInt32:
      return sizeof(uint32_t);
    case kTfLiteUInt16:
      return sizeof(uint16_t);
    default:
      throw std::runtime_error("Tensor DataType " + std::to_string(static_cast<int>(dataType)) +
                               " is not supported!");
  }
}

int getTensorTotalLength(const TfLiteTensor* tensor) {
  int dimensions = TfLiteTensorNumDims(tensor);
  if (dimensions < 1)
    return 0;
  int size = 1;
  for (int32_t i = 0; i < dimensions; i++) {
    size *= TfLiteTensorDim(tensor, i);
  }
  return size;
}

std::shared_ptr<TfLiteDelegate> getCoreMLDelegate() {
#ifdef __APPLE__
#if FAST_TFLITE_ENABLE_CORE_ML
  TfLiteCoreMlDelegateOptions delegateOptions;
  TfLiteDelegate* coreMlDelegate = TfLiteCoreMlDelegateCreate(&delegateOptions);
  if (coreMlDelegate == nullptr) {
    throw std::runtime_error("TFLite: Failed to create CoreML delegate!");
  }
  return std::shared_ptr<TfLiteDelegate>(coreMlDelegate, TfLiteCoreMlDelegateDelete);
#else // FAST_TFLITE_ENABLE_CORE_ML
  throw std::runtime_error("The CoreML Delegate (\"core-ml\") is not enabled! "
                           "Set `$EnableCoreMLDelegate` to `true` in your Podfile, and rebuild.");
#endif
#else // __APPLE__
  throw std::runtime_error(
      "The CoreML Delegate (\"core-ml\") is only supported on Apple Platforms!");
#endif
}

std::shared_ptr<TfLiteDelegate> getMetalDelegate() {
  throw std::runtime_error("Metal Delegate is not yet supported!");
}

std::shared_ptr<TfLiteDelegate> getNNAPIDelegate() {
#ifdef ANDROID
  TfLiteNnapiDelegateOptions delegateOptions = TfLiteNnapiDelegateOptionsDefault();
  TfLiteDelegate* nnapiDelegate = TfLiteNnapiDelegateCreate(&delegateOptions);
  if (nnapiDelegate == nullptr) {
    throw std::runtime_error("TFLite: Failed to create NNAPI delegate!");
  }
  return std::shared_ptr<TfLiteDelegate>(nnapiDelegate, TfLiteNnapiDelegateDelete);
#else // ANDROID
  throw std::runtime_error("The NNAPI Delegate (\"nnapi\") is only supported on Android!");
#endif
}

std::shared_ptr<TfLiteDelegate> getAndroidGPUDelegate() {
#ifdef ANDROID
  TfLiteGpuDelegateOptionsV2 delegateOptions = TfLiteGpuDelegateOptionsV2Default();
  TfLiteDelegate* gpuDelegate = TfLiteGpuDelegateV2Create(&delegateOptions);
  if (gpuDelegate == nullptr) {
    throw std::runtime_error("TFLite: Failed to create Android GPU delegate!");
  }
  return std::shared_ptr<TfLiteDelegate>(gpuDelegate, TfLiteGpuDelegateV2Delete);
#else // ANDROID
  throw std::runtime_error(
      "The Android GPU Delegate (\"android-gpu\") is only supported on Android!");
#endif
}

std::shared_ptr<TfLiteDelegate> getXNNPACKDelegate() {
#ifdef ANDROID
  TfLiteXNNPackDelegateOptions delegateOptions = TfLiteXNNPackDelegateOptionsDefault();
  delegateOptions.num_threads = getXNNPACKThreadCount();
  TfLiteDelegate* xnnpackDelegate = TfLiteXNNPackDelegateCreate(&delegateOptions);
  if (xnnpackDelegate == nullptr) {
    throw std::runtime_error("TFLite: Failed to create XNNPACK delegate!");
  }
  return std::shared_ptr<TfLiteDelegate>(xnnpackDelegate, TfLiteXNNPackDelegateDelete);
#else // ANDROID
  throw std::runtime_error("The XNNPACK Delegate (\"xnnpack\") is only supported on Android!");
#endif
}

} // namespace margelo::nitro::tflite
