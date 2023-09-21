cmake_minimum_required(VERSION 3.20.0)
project(cherry LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
endif()

set(CHERRY_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(CHERRY_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CHERRY_SOURCE_DIR}/include)
include_directories(${CHERRY_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_SRC_DIR ${CHERRY_SOURCE_DIR}/llvm-project CACHE STRING "The LLVM src directory")

add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
    message(STATUS "Enabling Python API")
    add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(cherry-opt)
add_subdirectory(cherry-plugin)
add_subdirectory(cherry-translate)
add_subdirectory(tools)
add_subdirectory(unittests)
