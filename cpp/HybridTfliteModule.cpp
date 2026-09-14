#include "HybridTfliteModule.hpp"
#include "TfliteHelpers.hpp"

#include <memory>
#include <utility>
#include <vector>

#if defined(ANDROID)
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

/**
 * Return a Hardware accelerated delegate, or throws
 * if the given delegate type is not available.
 */
TfLiteDelegate* getDelegate(TensorflowModelDelegate delegateType) {
  switch (delegateType) {
    case TensorflowModelDelegate::CORE_ML:
      return getCoreMLDelegate();
    case TensorflowModelDelegate::METAL:
      return getMetalDelegate();
    case TensorflowModelDelegate::NNAPI:
      return getNNAPIDelegate();
    case TensorflowModelDelegate::ANDROID_GPU:
      return getAndroidGPUDelegate();
  }
  throw std::runtime_error("Unknown Delegate \"" + std::to_string(static_cast<int>(delegateType)) +
                           "\"!");
}

/**
 * TFLite's C API does not transfer delegate ownership to the interpreter: the
 * caller must keep a delegate alive for the interpreter's lifetime and free it
 * afterwards with the delegate's own delete function.
 */
struct DelegateDeleter {
  TensorflowModelDelegate delegateType;

  void operator()(TfLiteDelegate* delegate) const {
    switch (delegateType) {
#if defined(__APPLE__) && FAST_TFLITE_ENABLE_CORE_ML
      case TensorflowModelDelegate::CORE_ML:
        TfLiteCoreMlDelegateDelete(delegate);
        return;
#endif
#if defined(ANDROID)
      case TensorflowModelDelegate::ANDROID_GPU:
        TfLiteGpuDelegateV2Delete(delegate);
        return;
      case TensorflowModelDelegate::NNAPI:
        TfLiteNnapiDelegateDelete(delegate);
        return;
#endif
      default:
        // getDelegate() throws for every other type on this platform.
        return;
    }
  }
};
using OwnedDelegate = std::unique_ptr<TfLiteDelegate, DelegateDeleter>;

std::shared_ptr<HybridTfliteModelSpec>
HybridTfliteModule::createModel(const std::shared_ptr<ArrayBuffer>& modelData,
                                const std::vector<TensorflowModelDelegate>& delegates) {
  const std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)> model(
      TfLiteModelCreate(modelData->data(), modelData->size()), TfLiteModelDelete);
  if (model == nullptr) {
    throw std::runtime_error("Failed to create TFLite model from data!");
  }

  // Configure interpreter via options
  const std::unique_ptr<TfLiteInterpreterOptions, decltype(&TfLiteInterpreterOptionsDelete)>
      options(TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete);
  if (options == nullptr) {
    throw std::runtime_error("TFLite: Failed to create interpreter options!");
  }

  // Add all hardware accelerated delegates (e.g. GPU, NPU, ...)
  // if any. The default CPU delegate will always be available.
  std::vector<TensorflowModelDelegate> effectiveDelegates;
  std::vector<OwnedDelegate> ownedDelegates;
  effectiveDelegates.reserve(delegates.size());
  ownedDelegates.reserve(delegates.size());
  for (const TensorflowModelDelegate& delegateType : delegates) {
    OwnedDelegate delegate(getDelegate(delegateType), DelegateDeleter{delegateType});
    if (delegate == nullptr) {
      // e.g. CoreML on devices without a Neural Engine — fall back to CPU
      // instead of registering a null delegate with the interpreter.
      continue;
    }
    TfLiteInterpreterOptionsAddDelegate(options.get(), delegate.get());
    effectiveDelegates.push_back(delegateType);
    ownedDelegates.push_back(std::move(delegate));
  }

  TfLiteInterpreter* rawInterpreter = TfLiteInterpreterCreate(model.get(), options.get());
  if (rawInterpreter == nullptr) {
    // `ownedDelegates` frees the delegates on unwind.
    throw std::runtime_error("Failed to create TFLite interpreter!");
  }
  // The delegates travel with the interpreter and are freed right after it,
  // so they can never be deleted while the interpreter still uses them.
  const std::shared_ptr<TfLiteInterpreter> interpreter(
      rawInterpreter,
      [modelData, ownedDelegates = std::move(ownedDelegates)](TfLiteInterpreter* value) mutable {
        TfLiteInterpreterDelete(value);
        ownedDelegates.clear();
      });

  // Wrap in HybridTfliteModel — stores shared_ptr<ArrayBuffer> to keep model data bytes alive.
  // Only the delegates that were actually registered are reported via `getDelegates()`.
  return std::make_shared<HybridTfliteModel>(interpreter, modelData, effectiveDelegates);
}

} // namespace margelo::nitro::tflite
