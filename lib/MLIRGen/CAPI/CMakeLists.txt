add_mlir_public_c_api_library(CherryCAPI
    Dialects.cpp
    LINK_LIBS PUBLIC
    cherryMLIRGen
)
