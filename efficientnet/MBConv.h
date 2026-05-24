#ifndef MB_CONV_H
#define MB_CONV_H

#include <vector>
#include "SEBlock.h"

class MBConv {
public:
    int channels;

    std::vector<std::vector<float>> dwKernel;
    std::vector<std::vector<float>> pwWeights;

    SEBlock se;

    MBConv(int channels);

    std::vector<float> forward(
        const std::vector<std::vector<float>>& input
    );

private:
    static float relu(float x);
};

#endif