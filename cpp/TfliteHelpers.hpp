#pragma once

#include "TensorDataType.hpp"
#include <memory>
#include <string>

#if defined(ANDROID)
#include <tflite/c/c_api.h>
#elif defined(__APPLE__)
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#else
#error "Invalid Platform!"
#endif

namespace margelo::nitro::tflite {

std::string tfLiteStatusToString(TfLiteStatus status);
TensorDataType getTensorDataType(TfLiteType dataType);
size_t getTFLTensorDataTypeSize(TfLiteType dataType);
int getTensorTotalLength(const TfLiteTensor* tensor);

std::shared_ptr<TfLiteDelegate> getCoreMLDelegate();
std::shared_ptr<TfLiteDelegate> getMetalDelegate();
std::shared_ptr<TfLiteDelegate> getNNAPIDelegate();
std::shared_ptr<TfLiteDelegate> getAndroidGPUDelegate();
std::shared_ptr<TfLiteDelegate> getXNNPACKDelegate();

} // namespace margelo::nitro::tflite
