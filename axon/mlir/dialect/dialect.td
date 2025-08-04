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
}

class AxonDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AxonDialect, name, traits> {
  let mnemonic = typeMnemonic;
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

def Axon_AccumulateOp : AxonDialect_Op<"accumulate"> {
  let arguments = (ins AnyMemRef:$lhs, AnyTensor:$rhs);
  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs)  attr-dict";
}

def Axon_StoreOp : AxonDialect_Op<"store"> {
  let arguments = (ins AnyMemRef:$to, AnyTensor:$value);
  let assemblyFormat = "$to `,` $value `:` type($to) `,` type($value)  attr-dict";
}

def Axon_AddOp : AxonDialect_Op<"add"> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
}

def Axon_MulOp : AxonDialect_Op<"mul"> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), lhs, rhs);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
}

def Axon_TupleAccessOp : AxonDialect_Op<"tuple_access"> {
  let summary = "Access the tensor from a tuple.";
  let arguments = (ins AnyTuple:$input, I64Attr:$index);
  let results = (outs AnyType:$output);

  let hasCustomAssemblyFormat = 1;
}

#endif // AXON_DIALECT
