#include "dialect.h"

#include "Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "DialectTypeDefs.cpp.inc"

namespace axon {

auto AxonDialect::initialize() -> void { addTypes<ParameterType>(); }

auto AxonDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter& printer) const -> void {
  if (auto param_type = mlir::cast<ParameterType>(type)) {
    printer << "param<";

    if (not param_type.isDynamic()) {
      llvm::interleave(param_type.getShape(), printer, "x");
      printer << "x";
    } else {
      printer << "*";
    }

    printer << param_type.getElementType() << '>';
  }
}

auto AxonDialect::parseType(mlir::DialectAsmParser& parser) const
    -> mlir::Type {
  if (parser.parseKeyword("param") || parser.parseGreater()) {
    return {};
  }

  llvm::SmallVector<int64_t, 3> shape;
  auto is_dynamic = false;
  if (parser.parseDimensionList(shape)) {
    if (parser.parseStar()) {
      return {};
    }
    is_dynamic = true;
  }

  mlir::Type elementType;
  if (parser.parseType(elementType) || parser.parseGreater()) {
    return {};
  }

  if (is_dynamic) {
    return ParameterType::getDynamic(elementType);
  } else {
    return ParameterType::get(shape, elementType);
  }
}

auto ParameterTypeStorage::construct(mlir::TypeStorageAllocator& allocator,
                                     const KeyTy& key)
    -> ParameterTypeStorage* {
  auto [shape, type] = key;
  shape = allocator.copyInto(shape);
  return new (allocator.allocate<ParameterTypeStorage>())
      ParameterTypeStorage(shape, type);
}

auto ParameterType::get(llvm::ArrayRef<int64_t> shape, mlir::Type type)
    -> ParameterType {
  mlir::MLIRContext* context = type.getContext();
  return Base::get(context, shape, type);
}

auto ParameterType::getDynamic(mlir::Type type) -> ParameterType {
  mlir::MLIRContext* context = type.getContext();
  return Base::get(context, llvm::ArrayRef<int64_t>{}, type);
}

}  // namespace axon
