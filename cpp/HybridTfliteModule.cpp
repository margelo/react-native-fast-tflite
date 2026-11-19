#include "HybridTfliteModule.hpp"
#include "TfliteHelpers.hpp"

#include <NitroModules/Promise.hpp>
#include <memory>

#if defined(__APPLE__)
#include "AutoreleasePool.hpp"
#endif

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

/**
 * Builds the interpreter from the `size` bytes at `bytes`, which `modelData` owns and keeps alive
 * for the interpreter's lifetime. It never calls into `modelData` itself, so it may run on any
 * thread.
 */
static std::shared_ptr<HybridTfliteModelSpec>
createModelFromBytes(const std::shared_ptr<ArrayBuffer>& modelData, const uint8_t* bytes,
                     size_t size, const std::vector<TensorflowModelDelegate>& delegates) {
  const std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)> model(
      TfLiteModelCreate(bytes, size), TfLiteModelDelete);
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

std::shared_ptr<HybridTfliteModelSpec>
HybridTfliteModule::createModel(const std::shared_ptr<ArrayBuffer>& modelData,
                                const std::vector<TensorflowModelDelegate>& delegates) {
  return createModelFromBytes(modelData, modelData->data(), modelData->size(), delegates);
}

std::shared_ptr<Promise<std::shared_ptr<HybridTfliteModelSpec>>>
HybridTfliteModule::createModelAsync(const std::shared_ptr<ArrayBuffer>& modelData,
                                     const std::vector<TensorflowModelDelegate>& delegates) {
  // The worker thread must not call into `modelData`: a JS-owned ArrayBuffer may only be read on
  // the JS thread, and on Android a natively owned one wraps a `java.nio.ByteBuffer`, whose address
  // can only be read on a thread attached to the JVM. So resolve the bytes here, on the calling
  // thread, after copying a buffer that does not own its memory. TFLite reads them for the
  // interpreter's whole lifetime, and `ownedData` keeps them alive for as long.
  std::shared_ptr<ArrayBuffer> ownedData =
      modelData->isOwner() ? modelData : ArrayBuffer::copy(modelData);
  const uint8_t* bytes = ownedData->data();
  size_t size = ownedData->size();
  return Promise<std::shared_ptr<HybridTfliteModelSpec>>::async(
      [ownedData, bytes, size, delegates]() -> std::shared_ptr<HybridTfliteModelSpec> {
#if defined(__APPLE__)
        // A thread-pool thread never drains an autorelease pool of its own, so whatever
        // Core ML / Metal autorelease while the interpreter is built would otherwise live forever.
        std::shared_ptr<HybridTfliteModelSpec> model;
        runInAutoreleasePool(
            [&] { model = createModelFromBytes(ownedData, bytes, size, delegates); });
        return model;
#else
        return createModelFromBytes(ownedData, bytes, size, delegates);
#endif
      });
}

} // namespace margelo::nitro::tflite
