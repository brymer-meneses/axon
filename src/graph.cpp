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

export class Graph {
 public:
  auto create_tensor(xt::xarray<float> data, bool requires_grad) -> InstId {
    auto data_id = data_.emplace_back(data);
    auto grad_id = requires_grad ? DataId::Pending : DataId::Invalid;

    return insts_.emplace_back(Create{}, data_id, grad_id);
  }

  auto create_data(xt::xarray<float> data) -> DataId {
    return data_.emplace_back(data);
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

export struct GraphExecutor {
  virtual auto forward() -> void = 0;
  virtual auto backward(InstId inst_id) -> void = 0;
};

}  // namespace axon
