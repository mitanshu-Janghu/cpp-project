#include <iostream>
#include "optimizer.h"

int main() {
    Tensor w({1});
    Tensor g({1});

    w.data[0] = 10.0f;
    g.data[0] = 1.0f;

    std::vector<Tensor*> params = { &w };
    std::vector<Tensor> grads = { g };

    Adam adam(0.1f);

    for (int i = 0; i < 5; ++i) {
        adam.step(params, grads);
        std::cout << "Step " << i << " w = " << w.data[0] << "\n";
    }
}
