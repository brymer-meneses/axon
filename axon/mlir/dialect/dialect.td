#ifndef AXON_DIALECT
#define AXON_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

def AxonDialect : Dialect {
  let name = "axon";
  let cppNamespace = "axon::backend";
  let useDefaultTypePrinterParser = 1;
}

class AxonDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AxonDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ArrayRefInt64 : TypeParameter<"::llvm::ArrayRef<int64_t>", "Array of int"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

def ArrayRefTensorRefType : TypeParameter<"::llvm::ArrayRef<TensorRefType>", "Array of params"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

def Axon_TensorRefType : AxonDialect_Type<"TensorRef", "tref"> {
  let parameters = (ins "mlir::Type":$elementType, ArrayRefInt64:$shape, "bool":$requires_grad);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    auto isDynamic() const -> bool { return getShape().empty(); }
  }];
}

def Axon_TensorRefListType : AxonDialect_Type<"TensorRefList", "list"> {
  let parameters = (ins ArrayRefTensorRefType:$values);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    auto operator[](int32_t index) const -> const TensorRefType& { return getValues()[index]; }
  }];
}

class AxonDialect_Op<string mnemonic, list<Trait> traits = []> : Op<AxonDialect, mnemonic, traits>;

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

def Axon_GetDataOp : AxonDialect_Op<"get_data"> {
  let summary = "Extract the data tensor from a tensor";

  let arguments = (ins Axon_TensorRefType:$input);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$input), [{
      auto tensor_ref_type = input.getType().cast<axon::backend::TensorRefType>();
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

def Axon_SetDataOp : AxonDialect_Op<"set_data"> {
  let summary = "Set the data tensor from a tensor";

  let arguments = (ins Axon_TensorRefType:$tensor, AnyTensor:$value);
}

def Axon_SetGradOp : AxonDialect_Op<"set_grad"> {
  let summary = "Set the gradient tensor from a tensor";

  let arguments = (ins Axon_TensorRefType:$tensor, AnyTensor:$value);
}

def Axon_AccumulateGradOp : AxonDialect_Op<"accumulate_grad"> {
  let summary = "Add the corresponding tensor to the gradient of this tensor.";

  let arguments = (ins Axon_TensorRefType:$tensor, AnyTensor:$value);
}

def Axon_ListAccessOp : AxonDialect_Op<"list_access"> {
  let summary = "Access the tensor from a TensorRefList";
  let arguments = (ins Axon_TensorRefListType:$input, I64Attr:$index);
  let results = (outs Axon_TensorRefType:$output);

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` qualified(type($output))
  }];
}

#endif // AXON_DIALECT
