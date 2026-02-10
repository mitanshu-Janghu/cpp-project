#pragma once
#include "layer.h"

class ReLU : public Layer {
public:
    Node* forward(Node* input);
};
