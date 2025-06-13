module;

#include <cassert>
#include <cstdint>

export module axon.tensor;

namespace axon {

export class TensorId {
 public:
  static const TensorId Invalid;

  constexpr TensorId(int32_t value) : value_(value) {}

  constexpr auto is_valid() const -> bool { return value_ != -1; }

  constexpr auto value() const -> int32_t {
    assert(is_valid());
    return value_;
  }

 private:
  int32_t value_;
};

inline constexpr auto TensorId::Invalid = TensorId(-1);

export struct Tensor {
  bool requires_grad = false;
  bool is_optimizable = false;
  TensorId id;
};

}  // namespace axon
