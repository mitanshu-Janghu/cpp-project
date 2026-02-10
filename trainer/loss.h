#pragma once
#include "node.h"

class MSELoss {
public:
    Node* forward(Node* pred, Node* target);
};

class BCELoss {
public:
    Node* forward(Node* pred, Node* target);
};

class SoftmaxCELoss {
public:
    Node* forward(Node* logits, Node* target);
};

class SoftmaxCENode : public Node {
public:
    SoftmaxCENode(Node* logits, Node* target);
    void backward() override;

private:
    std::vector<float> softmax;
};
