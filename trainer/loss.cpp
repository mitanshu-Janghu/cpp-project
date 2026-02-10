#include "loss.h"

// MSE = mean((pred - target)^2)
Node* MSELoss::forward(Node* pred, Node* target) {

    Node* diff = new AddNode(
        pred,
        new MulNode(
            target,
            new LeafNode(Tensor({1}))  // placeholder for -1
        )
    );

    // Fix target * (-1)
    diff->parents[1]->value.data[0] = -1.0f;

    Node* sq = new MulNode(diff, diff);
    return sq;
}
