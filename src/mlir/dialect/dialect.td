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
  let useDefaultTypePrinterParser = 1;
}

class AxonDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AxonDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Shape : TypeParameter<"::llvm::ArrayRef<int64_t>", "Array of int"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

def Params : TypeParameter<"::llvm::ArrayRef<ParameterType>", "Array of params"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

def Axon_ParameterType : AxonDialect_Type<"Parameter", "param"> {
  let parameters = (ins "mlir::Type":$elementType, Shape:$shape, "bool":$requires_grad);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    auto isDynamic() const -> bool { return getShape().empty(); }
  }];
}

def Axon_ParameterListType : AxonDialect_Type<"ParameterList", "param_list"> {
  let parameters = (ins Params:$params);
  let hasCustomAssemblyFormat = 1;
}

class AxonDialect_Op<string mnemonic, list<Trait> traits = []> : Op<AxonDialect, mnemonic, traits>;

def Axon_GetDataOp : AxonDialect_Op<"data"> {
  let summary = "Extract the data tensor from a parameter";

  let arguments = (ins Axon_ParameterType:$param);
  let results = (outs AnyTensor);
}

def Axon_GetGradOp : AxonDialect_Op<"grad"> {
  let summary = "Extract the gradient tensor from a parameter";

  let arguments = (ins Axon_ParameterType:$param);
  let results = (outs AnyTensor);
}

def Axon_AccumulateGradOp : AxonDialect_Op<"accumulate_grad"> {
  let summary = "Add the corresponding tensor to the gradient of this parameter.";

  let arguments = (ins Axon_ParameterType:$param, AnyTensor:$value);
}

def Axon_ListAccessOp : AxonDialect_Op<"list_access"> {
  let summary = "Access parameter from a ParameterList";
  let arguments = (ins Axon_ParameterListType:$input, I64Attr:$index);
  let results = (outs Axon_ParameterType:$output);

  let hasCustomAssemblyFormat = 1;
}

#endif // AXON_DIALECT
