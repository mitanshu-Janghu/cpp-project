add_library(engine
    engine.cpp
    ann_model.cpp
    cnn_model.cpp
)


target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(engine PUBLIC tensor autodiff layers trainer)
