#pragma once
#include <vector>
#include <iostream>

class Tensor {
public:
    float* data;
    std::vector<int> shape;

    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    ~Tensor();

    int size() const;

    Tensor clone() const;

    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    void print() const;
};
