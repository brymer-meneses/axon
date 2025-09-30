#ifndef AXON_DIALECT
#define AXON_DIALECT

include "mlir/IR/EnumAttr.td"
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

  let hasConstantMaterializer = 1;
}

class AxonDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AxonDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ArrayRefInt64 : TypeParameter<"::llvm::ArrayRef<int64_t>", "Array of int"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}


class AxonDialect_Op<string mnemonic, list<Trait> traits = []> : Op<AxonDialect, mnemonic, traits>;


def Axon_ConstantOp : AxonDialect_Op<"constant", [Pure, ConstantLike]> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
  ];

  let hasFolder = 1;
  let hasCustomAssemblyFormat = 1;
}

def Axon_AccumulateOp : AxonDialect_Op<"accumulate", [
  MemoryEffectsOpInterface,
]> {
  let arguments = (ins AnyTensor:$sink, AnyTensor:$source);

  let assemblyFormat = "$sink `,` $source `:` qualified(type($sink)) `,` type($source) attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    auto getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) -> void {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), &getSinkMutable());
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSourceMutable());
    }
  }];

  let hasCanonicalizeMethod = 1;
}

def Axon_FillOp : AxonDialect_Op<"fill_like", [Pure]> {
  let arguments = (ins AnyAttr:$fill_value);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$fill_value `->` type($result) attr-dict";
  let hasFolder = 1;
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

  let hasFolder = 1;
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

  let hasFolder = 1;
}

def Axon_SubOp : AxonDialect_Op<"sub", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `->` type($result) attr-dict";

  let hasFolder = 1;
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

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Axon_ExpandDimsOp : AxonDialect_Op<"expand_dims", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, ArrayAttr:$mappings);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `:` type($input) `->` type($result) attr-dict";
}

def Axon_NegOp: AxonDialect_Op<"neg", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
  Involution // f(f(x)) = x
]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `->` type($result) attr-dict";

  let hasFolder = 1;
}

def Axon_ScalarMulOp: AxonDialect_Op<"scalar_mul", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
]> {
  let arguments = (ins AnyTensor:$input, AnyAttr:$scalar);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `*` $scalar `->` type($result) attr-dict";

  let hasFolder = 1;
}

def Axon_SqueezeOp : AxonDialect_Op<"squeeze", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, I64Attr:$dim);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `:` type($input) `->` type($result) attr-dict";
}

def Axon_UnsqueezeOp : AxonDialect_Op<"unsqueeze", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, I64Attr:$dim);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `:` type($input) `->` type($result) attr-dict";
}

def Axon_ReshapeOp : AxonDialect_Op<"reshape", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, DenseI64ArrayAttr:$target_shape);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` type($input) `->` type($result) attr-dict";

  let hasFolder = 1;
  let hasVerifier = 1;
}

def Axon_SumOp : AxonDialect_Op<"sum", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, I64Attr:$dim, BoolAttr:$keep_dims);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $dim `:` type($input) `->` type($result) attr-dict";
}

def Axon_MeanOp : AxonDialect_Op<"mean", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, I64Attr:$dim, BoolAttr:$keep_dims);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $dim `:` type($input) `->` type($result) attr-dict";
}

def Axon_ArgMaxOp : AxonDialect_Op<"argmax", [
  Pure,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, I64Attr:$dim, BoolAttr:$keep_dims);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input `,` $dim `:` type($input) `->` type($result) attr-dict";
}

def Axon_SoftmaxOp : AxonDialect_Op<"softmax", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, I64Attr:$dim);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $dim `:` type($input) `->` type($result) attr-dict";
}

def Axon_ReluOp : AxonDialect_Op<"relu", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
  Idempotent
]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `->` type($result) attr-dict";
}

def ComparePredicateAttr : I32EnumAttr<"ComparePredicate", "comparison predicate", [
  I32EnumAttrCase<"eq", 0>, 
  I32EnumAttrCase<"ne", 6>, 
  I32EnumAttrCase<"lt", 2>, 
  I32EnumAttrCase<"le", 3>,
  I32EnumAttrCase<"gt", 4>, 
  I32EnumAttrCase<"ge", 5>
]> {
  let cppNamespace = "axon";
}

def Axon_CompareOp : AxonDialect_Op<"compare", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType,
]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, ComparePredicateAttr:$predicate);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs $rhs $predicate`->` type($result) attr-dict";
}

def Axon_PowOp : AxonDialect_Op<"pow", [
  Pure,
  SameOperandsAndResultType,
  SameOperandsAndResultElementType
]> {
  let arguments = (ins AnyTensor:$input, F64Attr:$exponent);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `->` type($input) attr-dict";
}

def Axon_TransposeOp : AxonDialect_Op<"transpose", [
  Pure,
  SameOperandsAndResultElementType,
]> {
  let arguments = (ins AnyTensor:$input, I64Attr:$from, I64Attr:$to);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input `,` `(` $from `,` $to `)` type($input) `->` type($result) attr-dict";
  let hasCanonicalizer = 1;
}

#endif // AXON_DIALECT
