#pragma once

// Assertions used across Axon.
//
// - AXON_ASSERT(cond, ...): Hard invariant/contract. Debug: print +
// unreachable;
//   Release: unreachable (no printing). Use for UB risks
//   (index/null/type/shape, IR invariants) where continuing is invalid.
// - AXON_DCHECK(cond, ...): Debug-only sanity check. Debug: print +
// unreachable;
//   Release: no-op. Use for developer aids or when user errors are handled
//   elsewhere with exceptions.
// - AXON_UNREACHABLE(...): Code path must be impossible. Debug: print +
//   unreachable; Release: unreachable.

#include <exception>
#include <print>
#include <utility>

#if defined(ENABLE_DCHECK)
#define AXON_UNREACHABLE(...)                                              \
  do {                                                                     \
    std::print(stderr, "{}:{} Unreachable hit", __FILE_NAME__, __LINE__);  \
    __VA_OPT__(std::print(stderr, ": "); std::print(stderr, __VA_ARGS__);) \
    std::println(stderr, "");                                              \
    std::unreachable();                                                    \
  } while (0)
#else
#define AXON_UNREACHABLE(...) std::unreachable()
#endif

#if defined(ENABLE_DCHECK)
#define AXON_DCHECK(condition, ...)                                          \
  do {                                                                       \
    if (not(condition)) [[unlikely]] {                                       \
      std::print(stderr, "{}:{} DCHECK failed `" #condition "`",             \
                 __FILE_NAME__, __LINE__);                                   \
      __VA_OPT__(std::print(stderr, ": "); std::print(stderr, __VA_ARGS__);) \
      std::println(stderr, "");                                              \
      std::unreachable();                                                    \
    }                                                                        \
  } while (0)
#else
#define AXON_DCHECK(condition, ...) ((void)0)
#endif

#if defined(ENABLE_DCHECK)
#define AXON_ASSERT(condition, ...)                                          \
  do {                                                                       \
    if (not(condition)) [[unlikely]] {                                       \
      std::print(stderr, "{}:{} ASSERT failed `" #condition "`",             \
                 __FILE_NAME__, __LINE__);                                   \
      __VA_OPT__(std::print(stderr, ": "); std::print(stderr, __VA_ARGS__);) \
      std::println(stderr, "");                                              \
      std::unreachable();                                                    \
    }                                                                        \
  } while (0)
#else
#define AXON_ASSERT(condition, ...)    \
  do {                                 \
    if (not(condition)) [[unlikely]] { \
      std::unreachable();              \
    }                                  \
  } while (0)
#endif
