module;

#include <memory>
#include <variant>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"

export module axon.graph;

import axon.storage;
import axon.ids;
import axon.inst;

namespace axon {

struct Data {
  using Storage = xt::xarray<float>;
  std::unique_ptr<Storage> data;

  Data(Storage data) : data(std::make_unique<Storage>(data)) {}
};

struct BackwardBuilder {
  using InstStorage = Storage<InstId, Inst>;

  static auto build(const Add& op, Inst& inst, const InstStorage& storage)
      -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(AddBackward(), op.lhs_id);
    }
    if (rhs.requires_grad()) {
      inst.deps.emplace_back(AddBackward(), op.rhs_id);
    }
  };

  static auto build(const Mul& op, Inst& inst, const InstStorage& storage)
      -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(MulBackward(rhs.data_id), op.lhs_id);
    }

    if (rhs.requires_grad()) {
      inst.deps.emplace_back(MulBackward(lhs.data_id), op.rhs_id);
    }
  };

  static auto build(const MatMul& op, Inst& inst, const InstStorage& storage)
      -> void {
    const auto& lhs = storage.get(op.lhs_id);
    const auto& rhs = storage.get(op.rhs_id);

    if (lhs.requires_grad() or rhs.requires_grad()) {
      inst.grad_id = DataId::Pending;
    }

    if (lhs.requires_grad()) {
      inst.deps.emplace_back(MatMulBackwardL(rhs.data_id), op.lhs_id);
    }

    if (rhs.requires_grad()) {
      inst.deps.emplace_back(MatMulBackwardR(lhs.data_id), op.rhs_id);
    }
  };

  static auto build(const Create& create, Inst& inst, InstStorage& storage)
      -> void {};
};

export class Graph {
 public:
  auto create_tensor(xt::xarray<float> data, bool requires_grad) -> InstId {
    auto data_id = data_.emplace_back(data);
    auto grad_id = requires_grad ? DataId::Pending : DataId::Invalid;

    return insts_.emplace_back(Create{}, data_id, grad_id);
  }

  template <typename Op, typename... Args>
  auto apply_operation(Args... args) -> InstId {
    Op op{std::forward<Args>(args)...};

    Inst inst{op, DataId::Pending};

    BackwardBuilder::build(op, inst, insts_);
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

}  // namespace axon
