#ifndef TRAINER_H
#define TRAINER_H

#include <vector>
#include "Model.h"

struct Batch {
    std::vector<std::vector<float>> x;
    std::vector<std::vector<float>> y;
};

class Trainer {
public:
    void train(Model& model,
               Dataset& dataset,
               int epochs,
               float lr);
};

#endif