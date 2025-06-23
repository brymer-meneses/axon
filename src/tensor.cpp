module;

#include <memory>

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
  auto data() const -> Data { return graph_->data(inst_id_); }

 private:
  // The execution graph where this tensor lives.
  std::shared_ptr<Graph> graph_;

  // The instruction that encodes how this tensor is constructed.
  InstId inst_id_;
};

}  // namespace axon
