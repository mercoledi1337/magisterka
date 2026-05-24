#ifndef RESNET_H
#define RESNET_H

#include "../core/Model.h"
#include "ResidualBlock.h"

class ResNet : public Model {
public:
    ResidualBlock block1;
    ResidualBlock block2;

    std::vector<float> output;

    ResNet(int size);

    std::vector<float> forward(const std::vector<float>& x) override;

    void backward(
        const std::vector<std::vector<float>>& x,
        const std::vector<std::vector<float>>& y,
        float lr
    ) override;
};

#endif