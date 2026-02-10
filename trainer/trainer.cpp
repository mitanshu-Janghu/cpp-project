#include "trainer.h"
#include <iostream>

void Trainer::train(
    Model& model,
    const Tensor& X,
    const Tensor& Y,
    int epochs,
    float lr,
    int batch_size
) {
    int N = X.shape[0];   // number of samples
    MSELoss loss_fn;
    float base_lr = lr;
    SGD optim(lr);


    for (int e = 0; e < epochs; ++e) {

        // -------- Learning Rate Decay --------
        float current_lr = base_lr * (1.0f / (1.0f + 0.1f * e));
        optim.set_lr(current_lr);
        float epoch_loss = 0.0f;

        for (int i = 0; i < N; i += batch_size) {

            int end = std::min(i + batch_size, N);

            std::vector<Node*> param_nodes;
            model.parameters(param_nodes);

            // zero grads
            for (Node* p : param_nodes) {
                for (int i = 0; i < p->grad.size(); ++i) {
                    p->grad.data[i] = 0.0f;
                }
            }


            for (int j = i; j < end; ++j) {

                // extract one sample
                Tensor x_i({1, X.shape[1]});
                Tensor y_i({1, Y.shape[1]});

                for (int k = 0; k < X.shape[1]; ++k)
                    x_i.data[k] = X.data[j * X.shape[1] + k];

                for (int k = 0; k < Y.shape[1]; ++k)
                    y_i.data[k] = Y.data[j * Y.shape[1] + k];

                LeafNode* x = new LeafNode(x_i);
                LeafNode* y = new LeafNode(y_i);

                Node* pred = model.forward(x);
                Node* loss = loss_fn.forward(pred, y);

                backward_all(loss);
                epoch_loss += loss->value.data[0];
            }

            // optimizer step (batch update)
            std::vector<Tensor*> params;
            std::vector<Tensor> grads;

            for (Node* p : param_nodes) {
                params.push_back(&p->value);
                grads.push_back(p->grad);
            }

            // -------- Gradient Clipping --------
            const float clip_value = 1.0f;

            for (Tensor& g : grads) {
                for (int i = 0; i < g.size(); ++i) {
                    if (g.data[i] > clip_value)
                    g.data[i] = clip_value;
                else if (g.data[i] < -clip_value)
                    g.data[i] = -clip_value;
                }
            }


            optim.step(params, grads);
        }

        std::cout << "Epoch " << e
                  << " | Avg Loss: "
                  << epoch_loss / N
                  << "\n";
    }
}
