#pragma once

#include "HybridTfliteModelSpec.hpp"
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#if defined(ANDROID)
#include <tflite/c/c_api.h>
#elif defined(__APPLE__)
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#else
#error "Invalid Platform!"
#endif

namespace margelo::nitro::tflite {

class HybridTfliteModel : public HybridTfliteModelSpec {
public:
  explicit HybridTfliteModel(std::shared_ptr<TfLiteInterpreter> interpreter,
                             std::shared_ptr<ArrayBuffer> modelData,
                             std::vector<TensorflowModelDelegate> delegates);
  ~HybridTfliteModel() override = default;

  /**
   * Free the interpreter, its delegates and our reference to the model bytes
   * NOW, without waiting for every runtime's GC to drop its reference (worklet
   * runtimes may not GC for a long time, especially while backgrounded).
   * Thread-safe: blocks until an in-flight inference on another thread has
   * completed. Every later runSync/run/getInputs/getOutputs call throws a
   * catchable JS error. Idempotent. Called by Nitro when JS invokes
   * `model.dispose()`.
   */
  void dispose() override;

  // Properties (from HybridTfliteModelSpec)
  std::vector<TensorflowModelDelegate> getDelegates() override;
  std::vector<Tensor> getInputs() override;
  std::vector<Tensor> getOutputs() override;

  // Methods (from HybridTfliteModelSpec)
  std::vector<std::shared_ptr<ArrayBuffer>>
  runSync(const std::vector<std::shared_ptr<ArrayBuffer>>& input) override;
  std::shared_ptr<Promise<std::vector<std::shared_ptr<ArrayBuffer>>>>
  run(const std::vector<std::shared_ptr<ArrayBuffer>>& input) override;

private:
  void copyInputBuffers(const std::vector<std::shared_ptr<ArrayBuffer>>& input);
  void invoke();
  std::vector<std::shared_ptr<ArrayBuffer>> copyOutputBuffers();
  std::shared_ptr<ArrayBuffer> getOutputBufferForTensor(const TfLiteTensor* tensor);

  // Caller must hold _lifecycleMutex. A disposed model has released its
  // interpreter; std::runtime_error surfaces as a catchable JS error.
  void throwIfDisposed() const {
    if (_interpreter == nullptr) {
      throw std::runtime_error("TFLite: Model was disposed!");
    }
  }

private:
  std::shared_ptr<TfLiteInterpreter> _interpreter;
  std::vector<TensorflowModelDelegate> _delegates;
  std::shared_ptr<ArrayBuffer> _modelData;
  std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>> _outputBuffers;

  // Serializes inference against dispose(). Uncontended in normal operation
  // (one lock per inference, ~ns vs ~ms inference cost).
  std::mutex _lifecycleMutex;
};

} // namespace margelo::nitro::tflite
