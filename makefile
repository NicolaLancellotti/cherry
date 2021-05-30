CHERRY_BUILD_DIR = ./build
LLVM_BUILD_DIR = ./llvm-project/build
LLVM_INSTALL_DIR = ./llvm-install

ifeq ($(shell uname),Darwin)
	JOBS = $(shell sysctl -n hw.logicalcpu)
else
	JOBS = $(shell nproc)
endif

all:	generate_llvm_project \
		install_llvm \
		generate_cherry_project \
		build_cherry

clean:
	@echo "Clean"
	@rm -rdf $(LLVM_BUILD_DIR)
	@rm -rdf $(LLVM_INSTALL_DIR)
	@rm -rdf $(CHERRY_BUILD_DIR)

generate_llvm_project:
	@echo "Generate LLVM Project"
	@mkdir -p $(LLVM_BUILD_DIR)
	@cmake -G Ninja -S ./llvm-project/llvm -B $(LLVM_BUILD_DIR) \
		-DLLVM_ENABLE_PROJECTS=mlir \
		-DLLVM_TARGETS_TO_BUILD="X86" \
		-DCMAKE_BUILD_TYPE=Debug \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_INSTALL_UTILS=ON \
		-DLLVM_ENABLE_RTTI=ON \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR)

install_llvm:
	@echo "Install LLVM"
	@mkdir -p $(LLVM_INSTALL_DIR)
	@cmake --build $(LLVM_BUILD_DIR) --target install -- -j$(JOBS)

generate_cherry_project:
	@echo "Generate Cherry Project"
	@mkdir -p $(CHERRY_BUILD_DIR)
	@cmake -G Ninja -S . -B $(CHERRY_BUILD_DIR) \
		-DMLIR_DIR=$(LLVM_INSTALL_DIR)/lib/cmake/mlir \
		-DLLVM_EXTERNAL_LIT=$(CURDIR)/$(LLVM_BUILD_DIR)/bin/llvm-lit \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++

build_cherry:
	@echo "Build Cherry"
	@cmake --build $(CHERRY_BUILD_DIR) --target check-cherry mlir-doc -- -j$(JOBS)
