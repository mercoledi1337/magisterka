#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <vector>

class ConvLayer {
public:
    int kernelSize;
    int stride;
    int inputWidth;

    std::vector<std::vector<float>> kernel;
    float bias;

    std::vector<std::vector<float>> output;

    ConvLayer(int kernelSize, int stride, int inputWidth);

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& input
    );

    static float relu(float x);
};

#endif