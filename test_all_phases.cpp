#include <iostream>

// Phase 1
#include "tensor.h"
#include "node.h"
#include "graph.h"
#include "dense.h"
#include "relu.h"

// Phase 2
#include "engine.h"

int main() {

    std::cout << "==============================\n";
    std::cout << " GLOBAL INTEGRATION TEST\n";
    std::cout << " Phase 1 + Phase 2\n";
    std::cout << "==============================\n\n";

    // -----------------------------
    // PHASE 1 : TENSOR TEST
    // -----------------------------
    std::cout << "[Phase 1] Tensor test\n";

    Tensor A({2,2});
    Tensor B({2,2});

    for (int i = 0; i < 4; ++i) {
        A.data[i] = i + 1;   // 1 2 3 4
        B.data[i] = 2.0f;
    }

    Tensor C = A.add(B);
    Tensor D = A.mul(B);

    std::cout << "Add: ";
    C.print();

    std::cout << "Mul: ";
    D.print();

    // -----------------------------
    // PHASE 1 : AUTODIFF GRAPH TEST
    // -----------------------------
    std::cout << "\n[Phase 1] Autodiff test\n";

    Tensor t1({1});
    Tensor t2({1});
    t1.data[0] = 3.0f;
    t2.data[0] = 4.0f;

    LeafNode* n1 = new LeafNode(t1);
    LeafNode* n2 = new LeafNode(t2);

    Node* sum = new AddNode(n1, n2);
    backward_all(sum);

    std::cout << "d(n1): ";
    n1->grad.print();

    std::cout << "d(n2): ";
    n2->grad.print();

    // -----------------------------
    // PHASE 1 : LAYERS FORWARD TEST
    // -----------------------------
    std::cout << "\n[Phase 1] Dense + ReLU forward test\n";

    Tensor Xf({1,2});
    Xf.data[0] = 1.0f;
    Xf.data[1] = -2.0f;

    LeafNode* xf = new LeafNode(Xf);

    Dense dense_fwd(2, 2);
    ReLU relu_fwd;

    Node* yf = dense_fwd.forward(xf);
    yf = relu_fwd.forward(yf);

    std::cout << "Forward output: ";
    yf->value.print();

    // -----------------------------
    // PHASE 2 : FULL TRAINING TEST
    // -----------------------------
    std::cout << "\n[Phase 2] Engine training test\n";

    Tensor X({1,1});
    Tensor Y({1,1});

    X.data[0] = 1.0f;
    Y.data[0] = 2.0f;

    Engine engine;
    engine.fit(X, Y);

    std::cout << "\n==============================\n";
    std::cout << " ALL TESTS COMPLETED SUCCESSFULLY\n";
    std::cout << "==============================\n";

    return 0;
}