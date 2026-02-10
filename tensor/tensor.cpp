#include "tensor.h"
#include <cassert>
#include <cstring>

Tensor::Tensor() : data(nullptr) {}

Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {
    int total = 1;
    for (int d : shape) total *= d;
    data = new float[total]{0.0f};
}

Tensor::Tensor(const Tensor& other) : shape(other.shape) {
    int total = other.size();
    data = new float[total];
    std::memcpy(data, other.data, total * sizeof(float));
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data;
        shape = other.shape;
        int total = other.size();
        data = new float[total];
        std::memcpy(data, other.data, total * sizeof(float));
    }
    return *this;
}

Tensor::~Tensor() {
    delete[] data;
}

int Tensor::size() const {
    int total = 0;
    for (int d : shape) total *= d;
    return total;
}

Tensor Tensor::clone() const {
    return Tensor(*this);
}

Tensor Tensor::add(const Tensor& other) const {
    assert(size() == other.size());
    Tensor out(shape);
    for (int i = 0; i < size(); ++i)
        out.data[i] = data[i] + other.data[i];
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    assert(size() == other.size());
    Tensor out(shape);
    for (int i = 0; i < size(); ++i)
        out.data[i] = data[i] * other.data[i];
    return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // Only supports 2D matrices
    assert(shape.size() == 2 && other.shape.size() == 2);
    assert(shape[1] == other.shape[0]);

    int m = shape[0];
    int n = shape[1];
    int p = other.shape[1];

    Tensor out({m, p});

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += data[i * n + k] *
                       other.data[k * p + j];
            }
            out.data[i * p + j] = sum;
        }
    }
    return out;
}

void Tensor::print() const {
    for (int i = 0; i < size(); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}
