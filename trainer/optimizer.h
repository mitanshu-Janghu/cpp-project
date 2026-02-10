#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>
#include "tensor.h"

// ---------------- SGD ----------------
class SGD {
    float lr;
public:
    SGD(float learning_rate);

    void step(std::vector<Tensor*>& params,
              const std::vector<Tensor>& grads);

    void set_lr(float new_lr) {
        lr = new_lr;
    }
};

// ---------------- Adam ----------------
class Adam {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;

    std::unordered_map<Tensor*, Tensor> m;
    std::unordered_map<Tensor*, Tensor> v;

public:
    Adam(float lr = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f);

    void step(std::vector<Tensor*>& params,
              const std::vector<Tensor>& grads);
};
