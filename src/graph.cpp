module;

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"

export module axon.graph;

import axon.ids;
import axon.inst;
import axon.storage;

namespace axon {

export struct Data {
  using Storage = xt::xarray<float>;
  std::shared_ptr<Storage> storage;

  Data(Storage storage) : storage(std::make_shared<Storage>(storage)) {}
};

template <typename T>
struct BackwardBuilder {};

template <typename T>
concept HasBackwardBuilder =
    requires(T op, Inst& inst, const Storage<InstId, Inst>& storage) {
      BackwardBuilder<T>::build(op, inst, storage);
    };

export class Graph {
 public:
  auto create_tensor(xt::xarray<float> data, bool requires_grad) -> InstId {
    auto data_id = data_.emplace_back(data);
    auto grad_id = requires_grad ? DataId::Pending : DataId::Invalid;

    return insts_.emplace_back(insts::Create{}, data_id, grad_id);
  }

  auto create_data(xt::xarray<float> data) -> DataId {
    return data_.emplace_back(data);
  }

  template <typename Op, typename... Args>
    requires(HasBackwardBuilder<Op>)
  auto apply_operation(Args... args) -> InstId {
    Op op{std::forward<Args>(args)...};

    Inst inst{op, DataId::Pending};

    BackwardBuilder<Op>::build(op, inst, insts_);

    return insts_.push_back(inst);
  }

  auto requires_grad(InstId inst_id) const -> bool {
    return insts_.get(inst_id).requires_grad();
  }

  auto insts() -> Storage<InstId, Inst>& { return insts_; }
  auto insts() const -> const Storage<InstId, Inst>& { return insts_; }

  auto data() -> Storage<DataId, Data>& { return data_; }
  auto data() const -> const Storage<DataId, Data>& { return data_; }

 private:
  Storage<InstId, Inst> insts_;
  Storage<DataId, Data> data_;
};

export struct GraphExecutor {
  virtual auto forward() -> void = 0;
  virtual auto backward(InstId inst_id) -> void = 0;
};

template <>
struct BackwardBuilder<insts::Create> {
  static auto build(const insts::Create&, Inst&, const Storage<InstId, Inst>&)
      -> void {};
};

template <>
struct BackwardBuilder<insts::Add> {
  static auto build(const insts::Add& op, Inst& inst,
                    const Storage<InstId, Inst>& storage) -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(insts::AddBackward(), op.lhs_id);
    }
    if (rhs.requires_grad()) {
      inst.deps.emplace_back(insts::AddBackward(), op.rhs_id);
    }
  };
};

template <>
struct BackwardBuilder<insts::Mul> {
  static auto build(const insts::Mul& op, Inst& inst,
                    const Storage<InstId, Inst>& storage) -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(insts::MulBackward(rhs.data_id), op.lhs_id);
    }

    if (rhs.requires_grad()) {
      inst.deps.emplace_back(insts::MulBackward(lhs.data_id), op.rhs_id);
    }
  };
};

template <>
struct BackwardBuilder<insts::MatMul> {
  static auto build(const insts::MatMul& op, Inst& inst,
                    const Storage<InstId, Inst>& storage) -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(insts::MatMulBackwardL(rhs.data_id), op.lhs_id);
    }

    if (rhs.requires_grad()) {
      inst.deps.emplace_back(insts::MatMulBackwardR(lhs.data_id), op.rhs_id);
    }
  }
};

}  // namespace axon
