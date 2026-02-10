#pragma once
#include "node.h"

class MSELoss {
public:
    Node* forward(Node* pred, Node* target);
};
