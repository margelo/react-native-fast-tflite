#pragma once

#if defined(__APPLE__)

#include <functional>

namespace margelo::nitro::tflite {

/**
 * Runs `body` inside an Objective-C autorelease pool.
 *
 * Threads that live for the whole process (like Nitro's thread pool) never drain
 * one of their own, so without it every object autoreleased by `body` would stay
 * alive until the app exits. An exception thrown by `body` is rethrown after the
 * pool has drained.
 */
void runInAutoreleasePool(const std::function<void()>& body);

} // namespace margelo::nitro::tflite

#endif
