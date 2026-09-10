#include "HybridTfliteModule.hpp"
#include "TfliteHelpers.hpp"

#include <algorithm>
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
 * Return a delegate, or throws if the given delegate type is not available.
 */
std::shared_ptr<TfLiteDelegate> getDelegate(TensorflowModelDelegate delegateType) {
  switch (delegateType) {
    case TensorflowModelDelegate::CORE_ML:
      return getCoreMLDelegate();
    case TensorflowModelDelegate::METAL:
      return getMetalDelegate();
    case TensorflowModelDelegate::NNAPI:
      return getNNAPIDelegate();
    case TensorflowModelDelegate::ANDROID_GPU:
      return getAndroidGPUDelegate();
    case TensorflowModelDelegate::XNNPACK:
      return getXNNPACKDelegate();
  }
  throw std::runtime_error("Unknown Delegate \"" + std::to_string(static_cast<int>(delegateType)) +
                           "\"!");
}

void validateDelegateConfiguration(const std::vector<TensorflowModelDelegate>& delegates) {
  const bool usesXNNPACK = std::find(delegates.begin(), delegates.end(),
                                     TensorflowModelDelegate::XNNPACK) != delegates.end();
  if (!usesXNNPACK) {
    return;
  }
  if (delegates.size() > 1) {
    throw std::runtime_error(
        "TFLite: The XNNPACK delegate cannot be combined with other delegates!");
  }
}

std::shared_ptr<HybridTfliteModelSpec>
HybridTfliteModule::createModel(const std::shared_ptr<ArrayBuffer>& modelData,
                                const std::vector<TensorflowModelDelegate>& delegates) {
  validateDelegateConfiguration(delegates);

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

  // Add all requested delegates. The default CPU kernels stay available for
  // operators outside delegated partitions.
  std::vector<std::shared_ptr<TfLiteDelegate>> delegateOwners;
  delegateOwners.reserve(delegates.size());
  for (const TensorflowModelDelegate& delegateType : delegates) {
    std::shared_ptr<TfLiteDelegate> delegate = getDelegate(delegateType);
    TfLiteInterpreterOptionsAddDelegate(options.get(), delegate.get());
    delegateOwners.push_back(std::move(delegate));
  }

  TfLiteInterpreter* rawInterpreter = TfLiteInterpreterCreate(model.get(), options.get());
  if (rawInterpreter == nullptr) {
    throw std::runtime_error("Failed to create TFLite interpreter!");
  }
  const std::shared_ptr<TfLiteInterpreter> interpreter(
      rawInterpreter,
      [modelData, delegateOwners = std::move(delegateOwners)](TfLiteInterpreter* value) {
        (void)modelData;
        (void)delegateOwners;
        TfLiteInterpreterDelete(value);
      });

  // Wrap in HybridTfliteModel — stores shared_ptr<ArrayBuffer> to keep model data bytes alive
  return std::make_shared<HybridTfliteModel>(interpreter, modelData, delegates);
}

} // namespace margelo::nitro::tflite
