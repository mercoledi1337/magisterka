#ifndef POINTWISE_CONV_H
#define POINTWISE_CONV_H

#include <vector>

class PointwiseConv {
public:
    int inChannels;
    int outChannels;

    std::vector<std::vector<float>> weights;
    std::vector<float> bias;

    PointwiseConv(int inC, int outC);

    std::vector<float> forward(
        const std::vector<std::vector<float>>& input
    );
};

#endif