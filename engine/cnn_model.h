#pragma once
#include "model.h"
#include <iostream>

class CNNModel : public Model {
public:
    CNNModel() {}

    Node* forward(Node* x) override {
        std::cout << "[CNNModel] Forward not implemented yet\n";
        return x;
    }

    void parameters(std::vector<Node*>& params) override {
        // No parameters yet
    }
};
