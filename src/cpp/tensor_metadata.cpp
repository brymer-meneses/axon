export module axon.tensor_metadata;

import axon.ids;

namespace axon {

export struct TensorMetadata {
 public:
  static auto CreateComputed(bool requires_grad) -> TensorMetadata {
    return TensorMetadata(DataId::Invalid, DataId::Invalid, requires_grad,
                          /*is_computed=*/true);
  }

  static auto Create(DataId data_id, DataId grad_id, bool requires_grad)
      -> TensorMetadata {
    return TensorMetadata(data_id, grad_id, requires_grad,
                          /*is_computed=*/false);
  }

  auto MarkAsObserved() -> void { is_observed = true; }

 private:
  TensorMetadata(DataId data_id, DataId grad_id, bool requires_grad,
                 bool is_computed)
      : data_id(data_id),
        grad_id(grad_id),
        requires_grad(requires_grad),
        is_computed(is_computed) {}

 public:
  const DataId data_id = DataId::Invalid;
  const DataId grad_id = DataId::Invalid;

  const bool requires_grad = false;
  const bool is_computed = false;

  bool is_alive = true;
  bool is_observed = false;
};
}  // namespace axon
