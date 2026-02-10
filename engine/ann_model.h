#pragma once
#include "model.h"
#include "dense.h"
#include "relu.h"

class ANNModel : public Model {
private:
    Dense d1;
    ReLU r1;
    Dense d2;

public:
    ANNModel(int in_features, int hidden, int out_features);

    Node* forward(Node* x) override;
    void parameters(std::vector<Node*>& params) override;
};
