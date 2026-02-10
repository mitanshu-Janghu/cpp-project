#include <iostream>
#include "trainer\loss.h"
#include "graph.h"

int main() {

    Tensor logits({3});
    logits.data[0] = 2.0f;
    logits.data[1] = 1.0f;
    logits.data[2] = 0.1f;

    Tensor target({3});
    target.data[0] = 1.0f;  // class 0
    target.data[1] = 0.0f;
    target.data[2] = 0.0f;

    LeafNode* l = new LeafNode(logits);
    LeafNode* t = new LeafNode(target);

    SoftmaxCELoss loss_fn;
    Node* loss = loss_fn.forward(l, t);

    backward_all(loss);

    std::cout << "Loss: " << loss->value.data[0] << "\n";
    std::cout << "Gradients:\n";
    l->grad.print();
}
