add_library(layers
    dense.cpp
    relu.cpp
)

target_include_directories(layers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(layers PUBLIC tensor autodiff)
