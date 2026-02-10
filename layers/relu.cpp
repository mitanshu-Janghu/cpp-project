#include "relu.h"

Node* ReLU::forward(Node* input) {
    return new ReLUNode(input);
}
