#include "dense.h"
#include <cstdlib>
#include <ctime>

static float rand_small() {
    return ((float)std::rand() / RAND_MAX - 0.5f) * 0.1f;
}

Dense::Dense(int in_features, int out_features) {
    static bool seeded = false;
    if (!seeded) {
        std::srand((unsigned)std::time(nullptr));
        seeded = true;
    }

    Tensor w({in_features, out_features});
    Tensor bias({1, out_features});

    for (int i = 0; i < w.size(); ++i)
        w.data[i] = rand_small();

    for (int i = 0; i < bias.size(); ++i)
        bias.data[i] = 0.0f;

    W = new LeafNode(w);
    b = new LeafNode(bias);
}

Node* Dense::forward(Node* input) {
    Node* z = new MatMulNode(input, W);
    Node* out = new AddNode(z, b);
    return out;
}
