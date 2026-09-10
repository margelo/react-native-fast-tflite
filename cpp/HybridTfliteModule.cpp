#include "HybridTfliteModule.hpp"
#include "TfliteHelpers.hpp"

#include <memory>

#if defined(ANDROID)
#include <tflite/c/c_api.h>
#elif defined(__APPLE__)
#include <TensorFlowLiteC/TensorFlowLiteC.h>
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
  for (const TensorflowModelDelegate& delegateType : delegates) {
    TfLiteDelegate* delegate = getDelegate(delegateType);
    TfLiteInterpreterOptionsAddDelegate(options.get(), delegate);
  }

  TfLiteInterpreter* rawInterpreter = TfLiteInterpreterCreate(model.get(), options.get());
  if (rawInterpreter == nullptr) {
    throw std::runtime_error("Failed to create TFLite interpreter!");
  }
  const std::shared_ptr<TfLiteInterpreter> interpreter(
      rawInterpreter, [modelData](TfLiteInterpreter* value) { TfLiteInterpreterDelete(value); });

  // Wrap in HybridTfliteModel — stores shared_ptr<ArrayBuffer> to keep model data bytes alive
  return std::make_shared<HybridTfliteModel>(interpreter, modelData, delegates);
}

} // namespace margelo::nitro::tflite
