#include "optimizer.h"

// ---------------- SGD ----------------
SGD::SGD(float learning_rate) : lr(learning_rate) {}

void SGD::step(std::vector<Tensor*>& params,
               const std::vector<Tensor>& grads) {

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* p = params[i];
        const Tensor& g = grads[i];

        for (int j = 0; j < p->size(); ++j) {
            p->data[j] -= lr * g.data[j];
        }
    }
}

// ---------------- Adam ----------------
Adam::Adam(float lr, float beta1, float beta2, float eps)
    : lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {}

void Adam::step(std::vector<Tensor*>& params,
                const std::vector<Tensor>& grads) {

    t++;

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* p = params[i];
        const Tensor& g = grads[i];

        // Initialize moments if first time
        if (m.find(p) == m.end()) {
            m[p] = Tensor(p->shape);
            v[p] = Tensor(p->shape);

            for (int j = 0; j < p->size(); ++j) {
                m[p].data[j] = 0.0f;
                v[p].data[j] = 0.0f;
            }
        }

        Tensor& mt = m[p];
        Tensor& vt = v[p];

        for (int j = 0; j < p->size(); ++j) {
            // Update biased first moment
            mt.data[j] = beta1 * mt.data[j] + (1.0f - beta1) * g.data[j];

            // Update biased second moment
            vt.data[j] = beta2 * vt.data[j] + (1.0f - beta2) * g.data[j] * g.data[j];

            // Bias correction
            float m_hat = mt.data[j] / (1.0f - std::pow(beta1, t));
            float v_hat = vt.data[j] / (1.0f - std::pow(beta2, t));

            // Parameter update
            p->data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}
