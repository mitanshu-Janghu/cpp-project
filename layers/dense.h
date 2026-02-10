#pragma once
#include "layer.h"

class Dense : public Layer {
private:
    Node* W;
    Node* b;

public:
    Dense(int in_features, int out_features);
    Node* forward(Node* input);

    Node* weights() { return W; }
    Node* bias() { return b; }
};
