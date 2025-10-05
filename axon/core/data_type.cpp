module;

#include <cstdint>
#include <type_traits>

#include "axon/base/macros.h"
#include "llvm/ADT/Hashing.h"
#include "nanobind/ndarray.h"

export module axon.core:data_type;

import axon.base;

namespace nb = nanobind;

namespace axon {

export class DataType {
 public:
  enum InternalType : u8 {
    Float32,
    Float64,
    Int1,
    Int32,
    Int64,
  };

  constexpr DataType(InternalType type) : type_(type) {}
  constexpr DataType() = default;

  auto operator==(DataType other) const -> bool { return type_ == other.type_; }

  auto getSizeInBytes() const -> size_t {
    switch (type_) {
      case DataType::Float32:
        return 4;
      case DataType::Float64:
        return 8;
      case DataType::Int1:
        return 1;
      case DataType::Int32:
        return 4;
      case DataType::Int64:
        return 8;
    }
  }

  auto asString() const -> std::string_view {
    switch (type_) {
      case DataType::Float32:
        return "float32";
      case DataType::Float64:
        return "float64";
      case DataType::Int1:
        return "bool";
      case DataType::Int32:
        return "int32";
      case DataType::Int64:
        return "int64";
    }
  }

  auto kind() const -> InternalType { return type_; }

  auto isFloatingPoint() const -> bool {
    return type_ == Float32 || type_ == Float64;
  }

  auto isInteger() const -> bool {
    switch (type_) {
      case Float32:
      case Float64:
        return false;
      case Int1:
      case Int32:
      case Int64:
        return true;
    }
  }

  template <Numeric T>
  auto isSameAs() const -> bool {
    return fromType<T>() == type_;
  }

  template <Numeric T>
  static consteval auto fromType() -> DataType {
    if constexpr (std::is_same_v<T, f32>) {
      return DataType::Float32;
    } else if constexpr (std::is_same_v<T, f64>) {
      return DataType::Float64;
    } else if constexpr (std::is_same_v<T, bool>) {
      return DataType::Int1;
    } else if constexpr (std::is_same_v<T, i32>) {
      return DataType::Int32;
    } else if constexpr (std::is_same_v<T, i64>) {
      return DataType::Int64;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      return DataType::Int32;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
      return DataType::Int64;
    } else {
      static_assert(false,
                    "Passed template parameter has no corresponding DataType");
    }
  }

  static auto fromDlPack(nb::dlpack::dtype dtype) -> DataType {
    if (dtype.code == static_cast<u8>(nanobind::dlpack::dtype_code::Float)) {
      if (dtype.bits == 32) {
        return DataType::Float32;
      } else if (dtype.bits == 64) {
        return DataType::Float64;
      }
    } else if (dtype.code ==
               static_cast<u8>(nanobind::dlpack::dtype_code::Int)) {
      if (dtype.bits == 1) {
        return DataType::Int1;
      } else if (dtype.bits == 32) {
        return DataType::Int32;
      } else if (dtype.bits == 64) {
        return DataType::Int64;
      }
    } else if (dtype.code ==
               static_cast<u8>(nanobind::dlpack::dtype_code::UInt)) {
      if (dtype.bits == 1) {
        return DataType::Int1;
      }
      if (dtype.bits == 32) {
        return DataType::Int32;
      }
      if (dtype.bits == 64) {
        return DataType::Int64;
      }
    } else if (dtype.code ==
               static_cast<u8>(nanobind::dlpack::dtype_code::Bool)) {
      return DataType::Int1;
    }

    AXON_UNREACHABLE("Unsupported dtype bits={} and code={}", dtype.bits,
                     dtype.code);
  }

 private:
  InternalType type_;
};

export auto hash_value(const DataType& data_type) -> llvm::hash_code {
  return llvm::hash_combine(data_type.kind());
}

}  // namespace axon

export template <>
struct std::formatter<axon::DataType> {
  constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

  auto format(const axon::DataType& data_type, std::format_context& ctx) const {
    auto out = ctx.out();
    return std::format_to(out, "{}", data_type.asString());
  }
};
