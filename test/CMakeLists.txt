llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/cherry/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/cherry/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/cherry/Unit/lit.cfg.py
)

set(CHERRY_TEST_DEPENDS
        FileCheck count not
        CherryUnitTests
        cherry-capi-test
        cherry-opt
        cherry-translate
        cherry-driver
        CherryPlugin
)

if(MLIR_ENABLE_BINDINGS_PYTHON)
    list(APPEND CHERRY_TEST_DEPENDS CherryPythonModules)
endif()

add_lit_testsuite(check-cherry "Running the cherry regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CHERRY_TEST_DEPENDS}
        )
set_target_properties(check-cherry PROPERTIES FOLDER "Tests")

add_lit_testsuites(cherry ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CHERRY_TEST_DEPENDS})

add_subdirectory(CAPI)
