#ifndef AXON_DIALECT
#define AXON_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"

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
    auto getTensorType() const -> mlir::Type { 
      return mlir::RankedTensorType::get(getShape(), getElementType());
    }
    auto getMemRefType() const -> mlir::Type { 
      return mlir::MemRefType::get(getShape(), getElementType());
    }
  }];

  let genMnemonicAlias = 1;
}

class AxonDialect_Op<string mnemonic, list<Trait> traits = []> : Op<AxonDialect, mnemonic, traits>;

def Axon_GetDataOp : AxonDialect_Op<"get_data"> {
  let summary = "Extract the data tensor from a tensor";

  let arguments = (ins Axon_TensorRefType:$input);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$input), [{
      auto tensor_ref_type = mlir::cast<axon::TensorRefType>(input.getType());
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

  let arguments = (ins Axon_TensorRefType:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `->` type($result)
  }];
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

def Axon_AccumulateOp : AxonDialect_Op<"accumulate", [
  MemoryEffectsOpInterface,
]> {
  let arguments = (ins AnyTensor:$sink, AnyTensor:$source);

  let assemblyFormat = "$sink `,` $source `:` qualified(type($sink)) `,` type($source) attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    auto getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) -> void{
      effects.emplace_back(mlir::MemoryEffects::Write::get());
      effects.emplace_back(mlir::MemoryEffects::Read::get());
    }
  }];
}

def Axon_FillOp : AxonDialect_Op<"fill_like", [Pure]> {
  let arguments = (ins F32Attr:$fill_value);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$fill_value `->` type($result) attr-dict";
}

def Axon_TupleAccessOp : AxonDialect_Op<"tuple_access", [
  Pure, 
  MemoryEffectsOpInterface, 
]> {
  let arguments = (ins AnyTuple:$input, I64Attr:$index);
  let results = (outs AnyType:$output);

  let builders = [
    OpBuilder<(ins "mlir::Value":$tuple, "int32_t":$index), [{
      auto tuple_casted = llvm::cast<mlir::TupleType>(tuple.getType());
      auto result_type = tuple_casted.getTypes()[index];
      build($_builder, $_state, result_type, tuple, index);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

def Axon_AddOp : AxonDialect_Op<"add", [
  Pure, 
  SameOperandsAndResultType, 
  SameOperandsAndResultElementType,
  Commutative
]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `->` type($result) attr-dict";

  let hasCanonicalizer = 1;
}

def Axon_MulOp : AxonDialect_Op<"mul", [
  Pure, 
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
  Commutative
]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `->` type($result) attr-dict";

  let hasCanonicalizer = 1;
}

def Axon_SubOp : AxonDialect_Op<"sub", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
  Commutative
]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `->` type($result) attr-dict";

  let hasCanonicalizer = 1;
}

def Axon_MatMulOp : AxonDialect_Op<"matmul", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins 
    AnyTensor:$lhs, 
    AnyTensor:$rhs, 
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$expanded_lhs_dims,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$expanded_rhs_dims
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Axon_ExpandDimsOp : AxonDialect_Op<"expand_dims", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$operand, ArrayAttr:$mappings);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$operand `:` type($operand) `->` type($result) attr-dict";
}

def Axon_NegOp: AxonDialect_Op<"neg", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
  Involution // f(f(x)) = x
]> {
  let arguments = (ins AnyTensor:$operand);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$operand `->` type($result) attr-dict";

  let hasCanonicalizer = 1;
}

def Axon_ScalarMulOp: AxonDialect_Op<"scalar_mul", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
]> {
  let arguments = (ins AnyTensor:$operand, F64Attr:$scalar);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$operand `*` $scalar `->` type($result) attr-dict";

  let hasCanonicalizer = 1;
}

def Axon_SqueezeOp : AxonDialect_Op<"squeeze", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$operand, I64Attr:$dim);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$operand `:` type($operand) `->` type($result) attr-dict";
}

def Axon_UnsqueezeOp : AxonDialect_Op<"unsqueeze", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$operand, I64Attr:$dim);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$operand `:` type($operand) `->` type($result) attr-dict";
}

def Axon_ReshapeOp : AxonDialect_Op<"reshape", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$operand, DenseI64ArrayAttr:$target_shape);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$operand `,` type($operand) `->` type($result) attr-dict";

  let hasVerifier = 1;
}

def Axon_SumOp : AxonDialect_Op<"sum", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$operand, I64Attr:$dim, BoolAttr:$keep_dims);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$operand `,` $dim `:` type($operand) `->` type($result) attr-dict";
}

def Axon_TransposeOp : AxonDialect_Op<"transpose", [
  Pure,
  SameOperandsAndResultElementType,
]> {
  let arguments = (ins AnyTensor:$operand, I64Attr:$from, I64Attr:$to);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$operand `,` `(` $from `,` $to `)` type($operand) `->` type($result) attr-dict";
  let hasCanonicalizer = 1;
}

#endif // AXON_DIALECT
