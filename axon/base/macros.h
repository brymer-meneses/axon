#pragma once

#include <exception>
#include <print>
#include <utility>

#if defined(ENABLE_DCHECK)
#define AXON_UNREACHABLE(...)                                                    \
  do {                                                                           \
    std::print(stderr, "{}:{} Unreachable hit", __FILE_NAME__, __LINE__);       \
    __VA_OPT__( std::print(stderr, ": "); std::print(stderr, __VA_ARGS__); )    \
    std::println(stderr, "");                                                   \
    std::terminate();                                                            \
  } while (0)
#else
#define AXON_UNREACHABLE(...) std::terminate()
#endif

#if defined(ENABLE_DCHECK)
#define AXON_DCHECK(condition, ...)                                              \
  do {                                                                           \
    if (not(condition)) [[unlikely]] {                                           \
      std::print(stderr, "{}:{} DCHECK failed `" #condition "`",              \
                 __FILE_NAME__, __LINE__);                                       \
      __VA_OPT__( std::print(stderr, ": "); std::print(stderr, __VA_ARGS__); )  \
      std::println(stderr, "");                                                 \
      std::terminate();                                                          \
    }                                                                            \
  } while (0)
#else
#define AXON_DCHECK(condition, ...) ((void)0)
#endif
