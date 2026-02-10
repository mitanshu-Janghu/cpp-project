add_library(tensor
    tensor.cpp
)

target_include_directories(tensor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
