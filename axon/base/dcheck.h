#pragma once

#include <exception>
#include <print>

#if !defined(NDEBUG)
#define AXON_DCHECK(condition, msg, ...)                                 \
  do {                                                                   \
    if (not(condition)) [[unlikely]] {                                   \
      std::println(stderr, "{}:{} DCHECK failed: " #condition " - " msg, \
                   __FILE_NAME__, __LINE__ __VA_OPT__(, ) __VA_ARGS__);  \
      std::terminate();                                                  \
    }                                                                    \
  } while (0)
#else
#define AXON_DCHECK(condition, msg, ...) ((void)0)
#endif
