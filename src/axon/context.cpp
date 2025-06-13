module;

#include "llvm/ADT/SmallVector.h"

export module axon.context;
import axon.op;
import axon.tensor;

namespace axon {

class TensorStorage {
 public:
  auto operator[](TensorId id) -> Tensor {
    assert(id.is_valid());
    return tensors_[id.value()];
  }

 private:
  llvm::SmallVector<Tensor> tensors_;
};

class Context {
 public:
  auto record_batch_matmul(TensorId left, TensorId right) -> void {}

  auto initialize_tensor() -> TensorId;

 private:
  TensorStorage storage_;
  llvm::SmallVector<TensorId> root_tensors_;
};

}  // namespace axon
