#pragma once

#include "HybridTfliteModel.hpp"
#include "HybridTfliteModuleSpec.hpp"

namespace margelo::nitro::tflite {

class HybridTfliteModule : public HybridTfliteModuleSpec {
public:
  HybridTfliteModule() : HybridObject(TAG) {}

  // Methods (from HybridTfliteModuleSpec)
  std::shared_ptr<HybridTfliteModelSpec>
  createModel(const std::shared_ptr<ArrayBuffer>& modelData,
              const std::vector<TensorflowModelDelegate>& delegates) override;
  std::shared_ptr<Promise<std::shared_ptr<HybridTfliteModelSpec>>>
  createModelAsync(const std::shared_ptr<ArrayBuffer>& modelData,
                   const std::vector<TensorflowModelDelegate>& delegates) override;
};

} // namespace margelo::nitro::tflite
