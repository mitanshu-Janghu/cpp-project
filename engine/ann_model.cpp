#include "ann_model.h"

ANNModel::ANNModel(int in_features, int hidden, int out_features)
    : d1(in_features, hidden),
      d2(hidden, out_features) {}

Node* ANNModel::forward(Node* x) {
    x = d1.forward(x);
    x = r1.forward(x);
    x = d2.forward(x);
    return x;
}

void ANNModel::parameters(std::vector<Node*>& params) {
    params.push_back(d1.weights());
    params.push_back(d1.bias());
    params.push_back(d2.weights());
    params.push_back(d2.bias());
}
