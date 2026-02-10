#pragma once
#include "model.h"
#include "loss.h"
#include "optimizer.h"
#include "graph.h"

class Trainer {
public:
    void train(Model& model,
               const Tensor& X,
               const Tensor& Y,
               int epochs,
               float lr,
               int batch_size = 1);
};
