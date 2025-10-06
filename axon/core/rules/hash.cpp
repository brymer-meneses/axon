module;

#include <print>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/HashBuilder.h"

export module axon.core:hash_rules;

import axon.base;

import :inst;
import :inst_kinds;
import :shape_rules;

namespace axon {

template <typename InstType>
concept BinaryInst = requires(InstType t) {
  t.lhs_id;
  t.rhs_id;
};

template <typename InstType>
concept UnaryInst = requires(InstType t) { t.input_id; };

template <typename T>
struct Hash;

// Hash simple binary insts
export template <BinaryInst T>
struct Hash<T> {
  static auto hash(const T& op, const IdMap<InstId, Shape>& shapes)
      -> llvm::hash_code {
    constexpr auto tag = Inst::tag<T>();
    llvm::ArrayRef<i64> lhs_shape(shapes.get(op.lhs_id)->get());
    llvm::ArrayRef<i64> rhs_shape(shapes.get(op.rhs_id)->get());

    return llvm::hash_combine(tag, lhs_shape, rhs_shape);
  }
};

// Hash simple unary insts
export template <UnaryInst T>
struct Hash<T> {
  static auto hash(const T& op, const IdMap<InstId, Shape>& shapes)
      -> llvm::hash_code {
    constexpr auto tag = Inst::tag<T>();
    llvm::ArrayRef<i64> operand_shape(shapes.get(op.input_id)->get());
    return llvm::hash_combine(tag, operand_shape);
  }
};

export template <>
struct Hash<insts::Reshape> {
  static auto hash(const insts::Reshape& op, const IdMap<InstId, Shape>& shapes)
      -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::Reshape>();

    llvm::ArrayRef<i64> operand_shape(shapes.get(op.input_id)->get());
    return llvm::hash_combine(tag, operand_shape);
  }
};

export template <>
struct Hash<insts::GetParameter> {
  static auto hash(const insts::GetParameter& op, const IdMap<InstId, Shape>&)
      -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::GetParameter>();

    return llvm::hash_combine(tag, op.param_id.value());
  }
};

template <typename InstType>
static auto handleReduceInst(const InstType& op,
                             const IdMap<InstId, Shape>& shapes) {
  constexpr auto tag = Inst::tag<InstType>();
  llvm::ArrayRef<i64> operand_shape(shapes.get(op.input_id)->get());
  return llvm::hash_combine(tag, operand_shape, op.keep_dims, op.axis);
}

export template <>
struct Hash<insts::Sum> {
  static auto hash(const insts::Sum& op, const IdMap<InstId, Shape>& shapes)
      -> llvm::hash_code {
    return handleReduceInst(op, shapes);
  }
};

export template <>
struct Hash<insts::ArgMax> {
  static auto hash(const insts::ArgMax& op, const IdMap<InstId, Shape>& shapes)
      -> llvm::hash_code {
    return handleReduceInst(op, shapes);
  }
};

export template <>
struct Hash<insts::Softmax> {
  static auto hash(const insts::Softmax& op, const IdMap<InstId, Shape>& shapes)
      -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::Softmax>();
    llvm::ArrayRef<i64> operand_shape(shapes.get(op.input_id)->get());
    return llvm::hash_combine(operand_shape, tag, op.axis);
  }
};

export template <>
struct Hash<insts::Compare> {
  static auto hash(const insts::Compare& op, const IdMap<InstId, Shape>& shapes)
      -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::Compare>();
    llvm::ArrayRef<i64> lhs_shape(shapes.get(op.lhs_id)->get());
    llvm::ArrayRef<i64> rhs_shape(shapes.get(op.rhs_id)->get());

    return llvm::hash_combine(tag, lhs_shape, rhs_shape, op.predicate);
  }
};

export template <>
struct Hash<insts::Unsqueeze> {
  static auto hash(const insts::Unsqueeze& op,
                   const IdMap<InstId, Shape>& shapes) -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::Unsqueeze>();

    llvm::ArrayRef<i64> operand_shape(shapes.get(op.input_id)->get());
    return llvm::hash_combine(tag, operand_shape, op.dim);
  }
};

export template <>
struct Hash<insts::Squeeze> {
  static auto hash(const insts::Squeeze& op, const IdMap<InstId, Shape>& shapes)
      -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::Squeeze>();

    llvm::ArrayRef<i64> operand_shape(shapes.get(op.input_id)->get());
    return llvm::hash_combine(tag, operand_shape, op.dim);
  }
};

export template <>
struct Hash<insts::Transpose> {
  static auto hash(const insts::Transpose& op,
                   const IdMap<InstId, Shape>& shapes) -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::Transpose>();

    llvm::ArrayRef<i64> operand_shape(shapes.get(op.input_id)->get());
    return llvm::hash_combine(tag, operand_shape, op.from, op.to);
  }
};

export template <>
struct Hash<insts::AccumulateGrad> {
  static auto hash(const insts::AccumulateGrad& op,
                   const IdMap<InstId, Shape>& shapes) -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::AccumulateGrad>();

    llvm::ArrayRef<i64> sink_shape(shapes.get(op.sink_id)->get());
    llvm::ArrayRef<i64> source_shape(shapes.get(op.source_id)->get());

    return llvm::hash_combine(tag, sink_shape, source_shape);
  }
};

export template <>
struct Hash<insts::AccumulateData> {
  static auto hash(const insts::AccumulateData& op,
                   const IdMap<InstId, Shape>& shapes) -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::AccumulateData>();

    llvm::ArrayRef<i64> sink_shape(shapes.get(op.sink_id)->get());
    llvm::ArrayRef<i64> source_shape(shapes.get(op.source_id)->get());

    return llvm::hash_combine(tag, sink_shape, source_shape);
  }
};

export template <>
struct Hash<insts::Constant> {
  static auto hash(const insts::Constant&, const IdMap<InstId, Shape>&)
      -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::Constant>();
    return llvm::hash_combine(tag);
  }
};

export template <>
struct Hash<insts::ScalarMul> {
  static auto hash(const insts::ScalarMul& op,
                   const IdMap<InstId, Shape>& shapes) -> llvm::hash_code {
    constexpr auto tag = Inst::tag<insts::ScalarMul>();
    llvm::ArrayRef<i64> operand_shape(shapes.get(op.input_id)->get());

    return llvm::hash_combine(tag, operand_shape, op.scalar);
  }
};

}  // namespace axon
