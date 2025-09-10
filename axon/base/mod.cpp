module;

#include <cstdint>

export module axon.base;

export import :index_base;
export import :storage;

export template <typename... Args>
struct match : Args... {
  using Args::operator()...;
};

export {
  using u8 = uint8_t;
  using u16 = uint16_t;
  using u32 = uint32_t;
  using u64 = uint64_t;

  using i8 = int8_t;
  using i16 = int16_t;
  using i32 = int32_t;
  using i64 = int64_t;

  using f32 = float;
  using f64 = double;
}
