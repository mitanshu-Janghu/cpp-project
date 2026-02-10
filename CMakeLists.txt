cmake_minimum_required(VERSION 3.16)

project(GOD_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(tensor)
add_subdirectory(autodiff)
add_subdirectory(layers)
add_subdirectory(trainer)
add_subdirectory(engine)

add_executable(test_adam test_adam.cpp)
target_link_libraries(test_adam tensor trainer)

