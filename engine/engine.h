#pragma once
#include "trainer.h"
#include "model.h"

class Engine {
public:
    void fit(const Tensor& X, const Tensor& Y);
};
