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

def Axon_ParameterType : DialectType<AxonDialect, CPred<"::llvm::isa<ParameterType>($_self)">, "Axon parameter type">;

def AxonDialect_Type : AnyTypeOf<[F64Tensor, Axon_ParameterType]>;

#endif // AXON_DIALECT
