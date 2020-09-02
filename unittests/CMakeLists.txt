add_custom_target(CherryUnitTests)
set_target_properties(CherryUnitTests PROPERTIES FOLDER "Cherry Tests")

function(add_cherry_unittest test_dirname)
    add_unittest(CherryUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Parse)