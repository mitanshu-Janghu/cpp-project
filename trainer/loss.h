#pragma once
#include "node.h"

// ---------------- MSE ----------------
class MSELoss {
public:
    Node* forward(Node* pred, Node* target);
};

// ---------------- BCE ----------------
class BCELoss {
public:
    Node* forward(Node* pred, Node* target);
};

// ---------------- Softmax + CE ----------------
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

