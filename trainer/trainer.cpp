#include "trainer.h"
#include <iostream>
#include <algorithm>

void Trainer::train(Model& model,
                    const Tensor& X,
                    const Tensor& Y,
                    int epochs,
                    float lr,
                    int batch_size) {

    int N = X.shape[0];
    MSELoss loss_fn;
    SGD optim(lr);
    float base_lr = lr;

    for (int e = 0; e < epochs; ++e) {
        optim.set_lr(base_lr / (1.0f + 0.1f * e));
        float epoch_loss = 0.0f;

        for (int i = 0; i < N; i += batch_size) {

            std::vector<Node*> param_nodes;
            model.parameters(param_nodes);

            for (Node* p : param_nodes)
                std::fill(p->grad.data.begin(), p->grad.data.end(), 0.0f);

            int end = std::min(i + batch_size, N);

            for (int j = i; j < end; ++j) {
                LeafNode* x = new LeafNode(X.row(j));
                LeafNode* y = new LeafNode(Y.row(j));

                Node* pred = model.forward(x);
                Node* loss = loss_fn.forward(pred, y);

                backward_all(loss);
                epoch_loss += loss->value.data[0];
            }

            std::vector<Tensor*> params;
            for (Node* p : param_nodes)
                params.push_back(&p->value);

            optim.step(params);
        }

        std::cout << "Epoch " << e
                  << " | Avg Loss: "
                  << epoch_loss / N << "\n";
    }
}
