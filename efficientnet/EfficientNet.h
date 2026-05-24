#ifndef EFFICIENT_NET_H
#define EFFICIENT_NET_H

#include "../core/Model.h"
#include "MBConv.h"

class EfficientNet : public Model {
public:
    MBConv block1;
    MBConv block2;

    std::vector<float> output;

    EfficientNet(int channels);

    std::vector<float> forward(const std::vector<float>& x) override;

    void backward(
        const std::vector<std::vector<float>>& x,
        const std::vector<std::vector<float>>& y,
        float lr
    ) override;
};

#endif