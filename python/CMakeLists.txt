include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_cherry`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_cherry.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CherryPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CherryPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_cherry"
  TD_FILE dialects/CherryOps.td
  SOURCES
    dialects/cherry.py
  DIALECT_NAME cherry)

declare_mlir_python_extension(CherryPythonSources.Extension
  MODULE_NAME _cherryDialects
  ADD_TO_PARENT CherryPythonSources
  SOURCES
        CherryExtension.cpp
  EMBED_CAPI_LINK_LIBS
    CherryCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(CherryPythonCAPI
  INSTALL_COMPONENT CherryPythonModules
  INSTALL_DESTINATION python_packages/cherry/mlir_cherry/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/cherry/mlir_cherry/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    CherryPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(CherryPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/cherry/mlir_cherry"
  INSTALL_PREFIX "python_packages/cherry/mlir_cherry"
  DECLARED_SOURCES
    CherryPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    CherryPythonCAPI
  )
