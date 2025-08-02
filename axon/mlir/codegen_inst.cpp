module;

#include "axon/base/dcheck.h"
#include "dialect/dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

export module axon.mlir:codegen_inst;

import axon.core;
import axon.base;

import :compilation_context;

namespace axon {

static auto codegen(insts::InitialGradient op, Context& context, InstId inst_id)
    -> void {}

static auto codegen(insts::SetCachedValue op, Context& context, InstId inst_id)
    -> void {}

static auto codegen(insts::GetCachedValue op, Context& context, InstId inst_id)
    -> void {}

static auto codegen(insts::AccumulateGrad op, Context& context, InstId inst_id)
    -> void {}

static auto codegen(insts::GetInput op, Context& context, InstId inst_id)
    -> void {}

static auto codegen(insts::LocalTensor, Context& context, InstId inst_id)
    -> void {}

static auto codegen(insts::Add op, Context& context, InstId inst_id) -> void {}

static auto codegen(insts::Mul op, Context& context, InstId inst_id) -> void {}

export auto codegen_inst(Context& context, InstId inst_id) -> void {}

}  // namespace axon
