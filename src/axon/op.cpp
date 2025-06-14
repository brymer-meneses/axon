module;

#include <variant>

export module axon.op;
import axon.tensor;

namespace axon {

export struct BatchMatMul {
  TensorId left;
  TensorId right;
};

export struct BatchScalarMul {
  double scalar;
  TensorId tensor;
};

export struct BatchMul {
  TensorId left;
  TensorId right;
};

export struct BatchAdd {
  TensorId left;
  TensorId right;
};

export struct BatchMinus {
  TensorId left;
  TensorId right;
};

export using Operation =
    std::variant<BatchMatMul, BatchScalarMul, BatchMul, BatchAdd, BatchMinus>;

}  // namespace axon
