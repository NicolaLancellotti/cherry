CHERRY_BUILD_DIR = ./build
LLVM_BUILD_DIR = ./llvm-project/build

ifeq ($(shell uname),Darwin)
	JOBS = $(shell sysctl -n hw.logicalcpu)
else
	JOBS = $(shell nproc)
endif

all:	generate_llvm_project \
		build_llvm \
		generate_cherry_project \
		build_cherry

clean:
	@echo "Clean"
	@rm -rdf $(LLVM_BUILD_DIR)
	@rm -rdf $(CHERRY_BUILD_DIR)

generate_llvm_project:
	@echo "Generate LLVM Project"
	@mkdir -p $(LLVM_BUILD_DIR)
	@cmake -G Ninja -S ./llvm-project/llvm -B $(LLVM_BUILD_DIR) \
		-DLLVM_ENABLE_PROJECTS=mlir \
		-DCMAKE_BUILD_TYPE=Debug \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_ENABLE_RTTI=ON \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

build_llvm:
	@echo "Build LLVM"
	@cmake --build $(LLVM_BUILD_DIR) -- -j$(JOBS)

generate_cherry_project:
	@echo "Generate Cherry Project"
	@mkdir -p $(CHERRY_BUILD_DIR)
	@cmake -G Ninja -S . -B $(CHERRY_BUILD_DIR) \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DLLVM_EXTERNAL_LIT=$(CURDIR)/$(LLVM_BUILD_DIR)/bin/llvm-lit \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

build_cherry:
	@echo "Build Cherry"
	@cmake --build $(CHERRY_BUILD_DIR) --target check-cherry mlir-doc -- -j$(JOBS)
