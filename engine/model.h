#pragma once
#include "node.h"

class Model {
public:
    virtual Node* forward(Node* x) = 0;
    virtual void parameters(std::vector<Node*>& params) = 0;
    virtual ~Model() {}
};
