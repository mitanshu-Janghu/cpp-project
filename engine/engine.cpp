#include "engine.h"
#include "ann_model.h"
#include "cnn_model.h"
#include <iostream>

void Engine::fit(const Tensor& X, const Tensor& Y) {

    std::cout << "[Engine] Auto-detecting model type...\n";

    int dims = X.shape.size();
    int input_dim = X.shape.back();
    int output_dim = Y.shape.back();

    Trainer trainer;

    // Case 1: Vector / tabular → ANN
    if (dims == 2) {
        std::cout << "[Engine] Selected ANN model\n";

        ANNModel model(input_dim, 8, output_dim);
        trainer.train(model, X, Y, 10, 0.01f, 2);
        return;
    }

    // Case 2: Image-like → CNN (stub)
    if (dims == 4) {
        std::cout << "[Engine] Selected CNN model (stub)\n";

        CNNModel model;
        std::cout << "[Engine] CNN training not implemented yet\n";
        return;
    }

    std::cout << "[Engine] Unsupported input shape\n";
}
