module;

#include "llvm/ADT/SmallVector.h"

export module axon.core:context;

import std;
import axon.base;

import :ids;

export namespace axon {

struct Data {
  std::unique_ptr<float> buffer;
  llvm::SmallVector<int64_t> shape;
};

class Context {
 public:
  auto create_empty(llvm::SmallVector<int64_t> shape) -> DataId {
    size_t size = std::ranges::fold_left(shape, 1, std::multiplies<>{});
    auto buffer = std::unique_ptr<float>(new float[size]);
    auto data_id = data_.emplace(std::move(buffer), std::move(shape));
    return data_id;
  }

 private:
  ValueStore<DataId, Data> data_;
};

}  // namespace axon
