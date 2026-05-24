#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <vector>
#include "Model.h"

class Evaluator {
public:
    float accuracy(Model& model,
                   const std::vector<std::vector<float>>& x,
                   const std::vector<std::vector<float>>& y);
};

#endif