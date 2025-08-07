#ifndef AXON_DIALECT
#define AXON_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

def AxonDialect : Dialect {
  let name = "axon";
  let cppNamespace = "axon";
  let useDefaultTypePrinterParser = 1;
}

class AxonDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AxonDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ArrayRefInt64 : TypeParameter<"::llvm::ArrayRef<int64_t>", "Array of int"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

def Axon_TensorRefType : AxonDialect_Type<"TensorRef", "tensor_ref"> {
  let parameters = (ins "mlir::Type":$elementType, ArrayRefInt64:$shape, "bool":$requires_grad);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    auto isDynamic() const -> bool { return getShape().empty(); }
  }];
}

class AxonDialect_Op<string mnemonic, list<Trait> traits = []> : Op<AxonDialect, mnemonic, traits>;

def Axon_GetDataOp : AxonDialect_Op<"get_data"> {
  let summary = "Extract the data tensor from a tensor";

  let arguments = (ins Axon_TensorRefType:$input);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$input), [{
      auto tensor_ref_type = input.getType().cast<axon::TensorRefType>();
      auto shape = tensor_ref_type.getShape();
      auto element_type = tensor_ref_type.getElementType();
      auto result_type = mlir::RankedTensorType::get(shape, element_type);
      build($_builder, $_state, result_type, input);
    }]>,
  ];

  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `->` type($result)
  }];
}

def Axon_GetGradOp : AxonDialect_Op<"get_grad"> {
  let summary = "Extract the gradient tensor from a tensor";

  let arguments = (ins Axon_TensorRefType:$tensor);
  let results = (outs AnyTensor);
}

def Axon_ConstantOp : AxonDialect_Op<"constant", [Pure]> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
}

def Axon_AccumulateGradOp : AxonDialect_Op<"accumulate_grad"> {
  let arguments = (ins Axon_TensorRefType:$lhs, AnyTensor:$rhs);
  let assemblyFormat = "$lhs `,` $rhs `:` qualified(type($lhs)) `,` type($rhs)  attr-dict";
}

def Axon_AddOp : AxonDialect_Op<"add"> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs);
    }]>,
  ];

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs)  `->` type($result) attr-dict";
}

def Axon_MulOp : AxonDialect_Op<"mul"> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs);
    }]>,
  ];

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs)  `->` type($result) attr-dict";
}

#endif // AXON_DIALECT
