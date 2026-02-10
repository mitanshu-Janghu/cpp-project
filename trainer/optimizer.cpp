#include "optimizer.h"
#include <cmath>

SGD::SGD(float lr) : lr(lr) {}

void SGD::set_lr(float new_lr) {
    lr = new_lr;
}

void SGD::step(const std::vector<Tensor*>& params) {
    for (Tensor* p : params) {
        for (int i = 0; i < p->size(); ++i)
            p->data[i] -= lr * p->grad.data[i];
    }
}

Adam::Adam(float lr, float b1, float b2, float eps)
    : lr(lr), beta1(b1), beta2(b2), eps(eps), t(0) {}

void Adam::step(const std::vector<Tensor*>& params) {
    t++;
    for (Tensor* p : params) {

        if (!m.count(p)) {
            m[p] = Tensor(p->shape);
            v[p] = Tensor(p->shape);
        }

        for (int i = 0; i < p->size(); ++i) {
            float g = p->grad.data[i];

            m[p].data[i] = beta1 * m[p].data[i] + (1 - beta1) * g;
            v[p].data[i] = beta2 * v[p].data[i] + (1 - beta2) * g * g;

            float m_hat = m[p].data[i] / (1 - std::pow(beta1, t));
            float v_hat = v[p].data[i] / (1 - std::pow(beta2, t));

            p->data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}
