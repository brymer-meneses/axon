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
  let cppNamespace = "axon";
  // let useDefaultTypePrinterParser = 1;
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

def Axon_TupleAccessOp : AxonDialect_Op<"tuple_access"> {
  let summary = "Access the tensor from a tuple.";
  let arguments = (ins AnyTuple:$input, I64Attr:$index);
  let results = (outs AnyType:$output);

  let hasCustomAssemblyFormat = 1;
}

#endif // AXON_DIALECT
