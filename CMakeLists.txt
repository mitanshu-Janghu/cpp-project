cmake_minimum_required(VERSION 3.16)

project(GOD_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(tensor)
add_subdirectory(autodiff)
add_subdirectory(layers)
add_subdirectory(trainer)
add_subdirectory(engine)

add_executable(test_bce test_bce.cpp)
target_link_libraries(test_bce tensor autodiff trainer)

add_executable(test_log test_log.cpp)
target_link_libraries(test_log tensor autodiff)

add_executable(test_softmax_ce test_softmax_ce.cpp)
target_link_libraries(test_softmax_ce tensor autodiff trainer)
