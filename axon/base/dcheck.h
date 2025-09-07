#pragma once

#include <exception>
#include <print>

#if !defined(ENABLE_DCHECK)
#define AXON_DCHECK(condition, ...)                                        \
  do {                                                                     \
    if (not(condition)) [[unlikely]] {                                     \
      if constexpr (sizeof(#__VA_ARGS__) > 1) {                            \
        std::println(stderr,                                               \
                     "{}:{} DCHECK failed `" #condition "`: " __VA_ARGS__, \
                     __FILE_NAME__, __LINE__);                             \
      } else {                                                             \
        std::println(stderr, "{}:{} DCHECK failed `" #condition "`",       \
                     __FILE_NAME__, __LINE__);                             \
      }                                                                    \
      std::terminate();                                                    \
    }                                                                      \
  } while (0)
#else
#define AXON_DCHECK(condition, ...) ((void)0)
#endif
