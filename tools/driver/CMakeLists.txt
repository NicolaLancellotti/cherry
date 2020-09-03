add_executable(cherry-driver driver.cpp)

target_link_libraries(cherry-driver
        LLVMSupport
        cherryDriver
        cherryParse
)