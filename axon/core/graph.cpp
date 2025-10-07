module;

#include <type_traits>

#include "axon/base/macros.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

export module axon.core:graph;

import axon.base;

import :shape_rules;
import :data_type;
import :ids;
import :storage;
import :inst;

namespace axon {

export struct Parameter {
  auto operator==(const Parameter& rhs) const -> bool = default;

  bool requires_grad = false;
  DataType data_type = DataType::Float32;
  InstId inst_id = InstId::None;
};

export class Graph {
 public:
  Graph() = default;

  auto operator==(const Graph& other) const -> bool = default;

  auto performBackward(InstId output_id, InstId grad_id) -> void;

  auto declareParam(llvm::ArrayRef<int64_t> shape, DataType data_type,
                    bool requires_grad) -> InstId;

  auto createConstant(Storage* constant) -> InstId;

  auto getShape(InstId inst_id) const -> ShapeRef {
    if (auto shape = shapes_.get(inst_id)) {
      return shape->get();
    };
    return {};
  }

  auto checkRequiresGrad(InstId inst_id) const -> bool {
    return gradients_.containsKey(inst_id);
  }

  auto createOp(Inst&& inst, bool emit_grad = true) -> InstId;

  auto merge(Graph& graph) -> void;

  auto hash() const -> u64;

  auto reset() -> void;

  auto getDataType(InstId inst_id) const -> DataType {
    auto data_type = data_types_.get(inst_id);
    AXON_ASSERT(data_type, "No data type registered for inst");
    return data_type->get();
  }

  auto gradients() -> IdStore<InstId, InstId>& { return gradients_; }
  auto gradients() const -> const IdStore<InstId, InstId>& {
    return gradients_;
  }

  auto insts() -> ValueStore<InstId, Inst>& { return insts_; }
  auto insts() const -> const ValueStore<InstId, Inst>& { return insts_; }

  auto constants() -> IdMap<InstId, Storage*>& { return constants_; }
  auto constants() const -> const IdMap<InstId, Storage*>& {
    return constants_;
  }

  auto parameters() -> ValueStore<ParamId, Parameter>& { return parameters_; }
  auto parameters() const -> const ValueStore<ParamId, Parameter>& {
    return parameters_;
  }

  auto shapes() -> IdMap<InstId, Shape>& { return shapes_; }
  auto shapes() const -> const IdMap<InstId, Shape>& { return shapes_; }

  auto data_types() -> IdMap<InstId, DataType>& { return data_types_; }
  auto data_types() const -> const IdMap<InstId, DataType>& {
    return data_types_;
  }

 private:
  auto inferShape(InstId inst_id) -> void;
  auto inferDataType(InstId inst_id) -> void;

 private:
  ValueStore<InstId, Inst> insts_;
  ValueStore<ParamId, Parameter> parameters_;

  IdMap<InstId, Storage*> constants_;
  IdMap<InstId, Shape> shapes_;
  IdMap<InstId, DataType> data_types_;

  IdStore<InstId, InstId> gradients_;

  llvm::SmallVector<InstId> returns_;
};

}  // namespace axon

export template <>
struct std::hash<axon::Graph> {
  auto operator()(const axon::Graph& graph) const -> size_t {
    return graph.hash();
  }
};
