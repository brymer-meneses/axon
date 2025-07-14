module;

#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:tensor;

import :ids;

export namespace axon {

class Module;

class TensorData {
 public:
  auto requires_grad() const -> bool {
    return grad_inst_id.has_value() or grad_inst_id == InstId::Pending;
  }

  auto is_local() const -> bool {
    return std::holds_alternative<LocalTensorData>(data_);
  }

  // This needs to be forward declared since this depends on methods for
  // `Module`. TODO: Maybe `Module` needs to be forward declared?
  auto shape() const -> llvm::ArrayRef<int64_t>;

  static auto create_input(Module* module, InstId tensor_inst_id,
                           bool requires_grad) -> TensorData {
    return TensorData{
        .grad_inst_id = requires_grad ? InstId::Pending : InstId::Invalid,
        .data_ = InputTensorData{
            .module = module,
            .tensor_inst_id = tensor_inst_id,
        }};
  }

  static auto create_local(llvm::SmallVector<int64_t> shape, bool requires_grad)
      -> TensorData {
    return TensorData{
        .grad_inst_id = requires_grad ? InstId::Pending : InstId::Invalid,
        .data_ = LocalTensorData{
            .shape = shape,
        }};
  }

 public:
  InstId grad_inst_id = InstId::Invalid;

  struct InputTensorData {
    Module* module;
    InstId tensor_inst_id;
  };

  struct LocalTensorData {
    llvm::SmallVector<int64_t> shape = {};
  };

  std::variant<InputTensorData, LocalTensorData> data_;
};

}  // namespace axon
