#pragma once
#include "node.h"
#include <vector>

inline void backward_all(Node* output) {
    output->grad.data[0] = 1.0f;

    std::vector<Node*> stack = {output};

    while (!stack.empty()) {
        Node* n = stack.back();
        stack.pop_back();
        n->backward();

        for (Node* p : n->parents)
            stack.push_back(p);
    }
}
