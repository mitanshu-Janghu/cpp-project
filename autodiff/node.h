#pragma once
#include "tensor.h"
#include <vector>

class Node {
public:
    Tensor value;
    Tensor grad;
    std::vector<Node*> parents;

    Node(const Tensor& val);
    virtual ~Node() {}

    virtual void backward() = 0;
};

class LeafNode : public Node {
public:
    LeafNode(const Tensor& val);
    void backward() override;
};

class AddNode : public Node {
public:
    AddNode(Node* a, Node* b);
    void backward() override;
};

class MulNode : public Node {
public:
    MulNode(Node* a, Node* b);
    void backward() override;
};

class MatMulNode : public Node {
public:
    MatMulNode(Node* a, Node* b);
    void backward() override;
};

class ReLUNode : public Node {
public:
    ReLUNode(Node* x);
    void backward() override;
};

class LogNode : public Node {
public:
    LogNode(Node* x);
    void backward() override;
};

class ExpNode : public Node {
public:
    ExpNode(Node* x);
    void backward() override;
};
