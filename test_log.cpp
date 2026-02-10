#include <iostream>
#include "node.h"
#include "graph.h"

int main() {
    Tensor x({1});
    x.data[0] = 2.0f;

    LeafNode* n = new LeafNode(x);
    Node* l = new LogNode(n);

    backward_all(l);

    std::cout << "log(2) = " << l->value.data[0] << "\n";
    std::cout << "d/dx log(x) = ";
    n->grad.print();
}
