#include "node.h"
#include <cassert>

// ---------- BASE ----------

Node::Node(const Tensor& val)
    : value(val), grad(val.shape) {
    for (int i = 0; i < grad.size(); ++i)
        grad.data[i] = 0.0f;
}

// ---------- LEAF ----------

LeafNode::LeafNode(const Tensor& val)
    : Node(val) {}

void LeafNode::backward() {
    // nothing
}

// ---------- ADD ----------

AddNode::AddNode(Node* a, Node* b)
    : Node(a->value.add(b->value)) {
    parents = {a, b};
}

void AddNode::backward() {
    for (int i = 0; i < grad.size(); ++i) {
        parents[0]->grad.data[i] += grad.data[i];
        parents[1]->grad.data[i] += grad.data[i];
    }
}

// ---------- MUL ----------

MulNode::MulNode(Node* a, Node* b)
    : Node(a->value.mul(b->value)) {
    parents = {a, b};
}

void MulNode::backward() {
    for (int i = 0; i < grad.size(); ++i) {
        parents[0]->grad.data[i] += parents[1]->value.data[i] * grad.data[i];
        parents[1]->grad.data[i] += parents[0]->value.data[i] * grad.data[i];
    }
}

// ---------- MATMUL ----------

MatMulNode::MatMulNode(Node* a, Node* b)
    : Node(a->value.matmul(b->value)) {
    parents = {a, b};
}

void MatMulNode::backward() {
    Node* A = parents[0];
    Node* B = parents[1];

    int m = A->value.shape[0];
    int n = A->value.shape[1];
    int p = B->value.shape[1];

    // dA = dOut * B^T
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < p; ++j) {
                sum += grad.data[i * p + j] *
                       B->value.data[k * p + j];
            }
            A->grad.data[i * n + k] += sum;
        }
    }

    // dB = A^T * dOut
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < m; ++i) {
                sum += A->value.data[i * n + k] *
                       grad.data[i * p + j];
            }
            B->grad.data[k * p + j] += sum;
        }
    }
}

// ---------- RELU ----------

ReLUNode::ReLUNode(Node* x)
    : Node(x->value) {
    parents = {x};

    for (int i = 0; i < value.size(); ++i)
        value.data[i] = value.data[i] > 0 ? value.data[i] : 0.0f;
}

void ReLUNode::backward() {
    for (int i = 0; i < grad.size(); ++i) {
        parents[0]->grad.data[i] +=
            (parents[0]->value.data[i] > 0) ? grad.data[i] : 0.0f;
    }
}
