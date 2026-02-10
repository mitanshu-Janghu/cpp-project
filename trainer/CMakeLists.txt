add_library(trainer
    loss.cpp
    optimizer.cpp
    trainer.cpp
)

target_include_directories(trainer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(trainer PUBLIC tensor autodiff layers engine)
