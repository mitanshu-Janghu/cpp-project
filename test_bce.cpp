#include <iostream>
#include "trainer\loss.h"
#include "autodiff\graph.h"

int main() {

    Tensor p({1});
    Tensor y({1});

    p.data[0] = 0.9f;
    y.data[0] = 1.0f;

    LeafNode* pred = new LeafNode(p);
    LeafNode* target = new LeafNode(y);

    BCELoss loss_fn;
    Node* loss = loss_fn.forward(pred, target);

    backward_all(loss);

    std::cout << "Loss: " << loss->value.data[0] << "\n";
    std::cout << "dPred: ";
    pred->grad.print();
}
