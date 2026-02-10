#pragma once
#include "node.h"

class Layer {
public:
    virtual Node* forward(Node* input) = 0;
    virtual ~Layer() {}
};
