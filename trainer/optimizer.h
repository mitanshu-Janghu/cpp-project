#pragma once
#include <vector>
#include <unordered_map>
#include "tensor.h"

class SGD {
    float lr;
public:
    explicit SGD(float lr);
    void step(const std::vector<Tensor*>& params);
    void set_lr(float new_lr);
};

class Adam {
    float lr, beta1, beta2, eps;
    int t;
    std::unordered_map<Tensor*, Tensor> m, v;

public:
    Adam(float lr = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f);

    void step(const std::vector<Tensor*>& params);
};
