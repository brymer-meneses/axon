module;

#include <memory>
#include <stdexcept>

export module axon.tensor;

import axon.graph;
import axon.inst;
import axon.ids;

namespace axon {

export class Tensor {
 public:
  Tensor(InstId inst_id, std::shared_ptr<Graph> graph)
      : inst_id_(inst_id), graph_(graph) {}

  auto requires_grad() const -> bool { return graph_->requires_grad(inst_id_); }
  auto inst_id() const -> InstId { return inst_id_; }
  auto data() const -> Data {
    const auto& inst = graph_->insts().get(inst_id_);
    if (inst.data_id == DataId::Pending) {
      throw std::runtime_error("Cannot call `data` on a lazy tensor.");
    } else if (inst.data_id == DataId::Invalid) {
      throw std::runtime_error("Cannot call `data` on an invalid tensor.");
    }

    return graph_->data().get(inst.data_id);
  }

 private:
  // The execution graph where this tensor lives.
  std::shared_ptr<Graph> graph_;

  // The instruction that encodes how this tensor is constructed.
  InstId inst_id_;
};

}  // namespace axon
