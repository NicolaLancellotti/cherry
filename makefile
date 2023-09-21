PROJECT_DIR=${shell cd .; pwd}
CHERRY_BUILD_DIR=./build
LLVM_COMMIT=llvmorg-17.0.1
LLVM_PYTHON_ENV=${HOME}/.venv/mlirdev
LLVM_SRC_DIR=${PROJECT_DIR}/llvm-project
LLVM_BUILD_DIR=${PROJECT_DIR}/llvm-project/build

.PHONY: all
all:	llvm \
		cherry

.PHONY: help
help:
	@echo "Targets:"
	@sed -nr 's/^.PHONY:(.*)/\1/p' ${MAKEFILE_LIST}

.PHONY: llvm
llvm: 	clone_llvm \
		checkout_llvm \
		generate_llvm_python_env \
		generate_llvm_project \
		build_llvm

.PHONY: cherry
cherry: generate_cherry_project \
		build_cherry

.PHONY: clean_llvm
clean_llvm:
	@echo "Clean LLVM"
	@rm -rdf ${LLVM_BUILD_DIR}

.PHONY: clean_cherry
clean_cherry:
	@echo "Clean Cherry"
	@rm -rdf ${CHERRY_BUILD_DIR}

.PHONY: clone_llvm
clone_llvm:
	@echo "Clone LLVM"
	-git clone https://github.com/llvm/llvm-project.git

.PHONY: checkout_llvm
checkout_llvm:
	@echo "Checkout LLVM"
	@cd ${LLVM_SRC_DIR} && git fetch && git checkout ${LLVM_COMMIT}

.PHONY: generate_llvm_python_env
generate_llvm_python_env:
	@echo "Generate LLVM Python Environment"
	@/usr/bin/python3 -m venv ${LLVM_PYTHON_ENV} && \
		source ${LLVM_PYTHON_ENV}/bin/activate && \
		python -m pip install --upgrade pip && \
		python -m pip install -r ${LLVM_SRC_DIR}/mlir/python/requirements.txt

.PHONY: generate_llvm_project
generate_llvm_project:
	@echo "Generate LLVM Project"
	@cmake -G Ninja -S ${LLVM_SRC_DIR}/llvm -B ${LLVM_BUILD_DIR} \
		-DLLVM_ENABLE_PROJECTS=mlir \
		-DLLVM_TARGETS_TO_BUILD=host \
		-DCMAKE_BUILD_TYPE=Debug \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_BUILD_TESTS=ON \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE=${LLVM_PYTHON_ENV}/bin/python3 \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

.PHONY: build_llvm
build_llvm:
	@echo "Build LLVM"
	@cmake --build ${LLVM_BUILD_DIR}

.PHONY: generate_cherry_project
generate_cherry_project:
	@echo "Generate Cherry Project"
	@mkdir -p ${CHERRY_BUILD_DIR}
	@cmake -G Ninja -S . -B ${CHERRY_BUILD_DIR} \
		-DLLVM_SRC_DIR=${LLVM_SRC_DIR} \
		-DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
		-DLLVM_EXTERNAL_LIT=${LLVM_BUILD_DIR}/bin/llvm-lit \
		-DCMAKE_BUILD_TYPE=Debug \
		-DPython3_EXECUTABLE=${LLVM_PYTHON_ENV}/bin/python3 \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

.PHONY: build_cherry
build_cherry:
	@echo "Build Cherry"
	@cmake --build ${CHERRY_BUILD_DIR} --target check-cherry mlir-doc

.PHONY: format
format:
	@find . -name "*.cpp" -or -name "*.h" | xargs clang-format -i
