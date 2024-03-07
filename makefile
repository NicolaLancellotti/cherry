# _____________________________________________________________________________
# Parameters

CHERRY_PRESET=debug

LLVM_COMMIT=llvmorg-18.1.0
LLVM_PRESET=release
LLVM_SRC_DIR=${PROJECT_DIR}/llvm-project
LLVM_BUILD_DIR=${LLVM_SRC_DIR}/build/${LLVM_PRESET}
LLVM_PYTHON_ENV=${HOME}/.venv/mlirdev

# _____________________________________________________________________________
# Paths

PROJECT_DIR=${shell cd .; pwd}
CHERRY_BUILD_DIR=./build

# _____________________________________________________________________________
# Targets

.PHONY: all
all:	llvm-all \
		cherry-all

.PHONY: help
help:
	@echo "Targets:"
	@sed -nr 's/^.PHONY:(.*)/\1/p' ${MAKEFILE_LIST}

define format
	@find ${1} -name "*.cpp" -or -name "*.h" | xargs clang-format -i
endef

.PHONY: format
format:
	@echo "Format"
	@$(call format, cherry-opt)
	@$(call format, cherry-plugin)
	@$(call format, cherry-translate)
	@$(call format, include)
	@$(call format, lib)
	@$(call format, test)
	@$(call format, tools)
	@$(call format, unittests)

# _____________________________________________________________________________
# Targets - LLVM

.PHONY: llvm-all
llvm-all: 	llvm-clone \
			llvm-checkout \
			llvm-generate-python-env \
			llvm-generate-project \
			llvm-build

.PHONY: llvm-clean
llvm-clean:
	@echo "LLVM - Clean"
	@rm -rdf ${LLVM_BUILD_DIR}

.PHONY: llvm-clone
llvm-clone:
	@echo "LLVM - Clone"
	-git clone https://github.com/llvm/llvm-project.git

.PHONY: llvm-checkout
llvm-checkout:
	@echo "LLVM - Checkout"
	@cd ${LLVM_SRC_DIR} && git fetch && git checkout ${LLVM_COMMIT}

.PHONY: llvm-generate-python-env
llvm-generate-python-env:
	@echo "LLVM - Generate Python Environment"
	@/usr/bin/python3 -m venv ${LLVM_PYTHON_ENV} && \
		source ${LLVM_PYTHON_ENV}/bin/activate && \
		python -m pip install --upgrade pip && \
		python -m pip install -r ${LLVM_SRC_DIR}/mlir/python/requirements.txt

.PHONY: llvm-generate-project
llvm-generate-project:
	@echo "LLVM - Generate Project"
	@cmake -G Ninja -S ${LLVM_SRC_DIR}/llvm -B ${LLVM_BUILD_DIR} \
		-DLLVM_ENABLE_PROJECTS=mlir \
		-DLLVM_TARGETS_TO_BUILD=host \
		-DCMAKE_BUILD_TYPE=${LLVM_PRESET} \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_BUILD_TESTS=ON \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE=${LLVM_PYTHON_ENV}/bin/python3 \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

.PHONY: llvm-build
llvm-build:
	@echo "LLVM - Build"
	@cmake --build ${LLVM_BUILD_DIR}

# _____________________________________________________________________________
# Targets - Cherry

.PHONY: cherry-all
cherry-all: cherry-generate-presets \
			cherry-generate-project \
			cherry-copy-compile-commands \
			cherry-build

.PHONY: cherry-clean
cherry-clean:
	@echo "Cherry - Clean"
	@rm -rdf ${CHERRY_BUILD_DIR}

.PHONY: cherry-generate-presets
cherry-generate-presets:
	@echo "Cherry - Generate Presets"
	@echo $$CMAKE_PRESETS_TEMPLATE > ./CMakeUserPresets.json

.PHONY: cherry-generate-project
cherry-generate-project:
	@echo "Cherry - Generate Project"
	@cmake -S ${PROJECT_DIR} --preset ${CHERRY_PRESET}

.PHONY: cherry-copy-compile-commands
cherry-copy-compile-commands:
	@echo "Cherry - Copy compile_commands.json"
	@cp ${PROJECT_DIR}/build/${CHERRY_PRESET}/compile_commands.json  ${PROJECT_DIR}/build

.PHONY: cherry-build
cherry-build:
	@echo "Cherry - Build"
	@cmake --build ${PROJECT_DIR}/build/${CHERRY_PRESET} --target check-cherry mlir-doc

# _____________________________________________________________________________
# Presets

define CMAKE_PRESETS_TEMPLATE
{
    "version": 3,
    "configurePresets": [
        {
            "name": "default",
            "hidden": true,
            "displayName": "Default configure preset",
            "description": "Default configure preset",
            "generator": "Ninja",
            "binaryDir": "./build/$${presetName}",
            "cacheVariables": {
				"LLVM_SRC_DIR": "${LLVM_SRC_DIR}",
				"MLIR_DIR": "${LLVM_BUILD_DIR}/lib/cmake/mlir",
				"LLVM_EXTERNAL_LIT": "${LLVM_BUILD_DIR}/bin/llvm-lit",
				"Python3_EXECUTABLE": "${LLVM_PYTHON_ENV}/bin/python3",
				"CMAKE_C_COMPILER": "/usr/bin/clang",
                "CMAKE_CXX_COMPILER": "/usr/bin/clang++",
				"CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
            }
        },
        {
            "name": "debug",
            "inherits": "default",
            "displayName": "Debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug"
            }
        },
        {
            "name": "release",
            "inherits": "default",
            "displayName": "Release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            }
        },
        {
            "name": "relWithDebInfo",
            "inherits": "default",
            "displayName": "RelWithDebInfo",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "RelWithDebInfo"
            }
        }
    ]
}
endef
export CMAKE_PRESETS_TEMPLATE
