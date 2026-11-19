#include "AutoreleasePool.hpp"

#include <exception>

namespace margelo::nitro::tflite {

void runInAutoreleasePool(const std::function<void()>& body) {
  std::exception_ptr failure;
  @autoreleasepool {
    try {
      body();
    } catch (...) {
      failure = std::current_exception();
    }
  }
  if (failure) {
    std::rethrow_exception(failure);
  }
}

} // namespace margelo::nitro::tflite
