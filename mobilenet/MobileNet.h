#ifndef MOBILENET_H
#define MOBILENET_H

#include "../core/Model.h"
#include "DepthwiseConv.h"
#include "PointwiseConv.h"

class MobileNet : public Model {
public:
    DepthwiseConv dw1;
    PointwiseConv pw1;

    std::vector<float> output;

    MobileNet();

    std::vector<float> forward(const std::vector<float>& x) override;

    void backward(
        const std::vector<std::vector<float>>& x,
        const std::vector<std::vector<float>>& y,
        float lr
    ) override;
};

#endif