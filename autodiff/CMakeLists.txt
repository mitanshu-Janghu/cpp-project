add_library(autodiff
    node.cpp
)

target_include_directories(autodiff PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(autodiff PUBLIC tensor)
