#ifndef CNN_H
#define CNN_H

#include "../core/Model.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "Flatten.h"

class CNN : public Model {
public:
    ConvLayer conv1;
    PoolLayer pool1;
    Flatten flatten;

    std::vector<float> fcWeights;
    std::vector<float> output;

    CNN();

    std::vector<float> forward(const std::vector<float>& x) override;

    void backward(
        const std::vector<std::vector<float>>& x,
        const std::vector<std::vector<float>>& y,
        float lr
    ) override;
};

#endif