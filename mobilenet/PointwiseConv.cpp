#include "PointwiseConv.h"
#include <random>

PointwiseConv::PointwiseConv(int inC, int outC)
    : inChannels(inC), outChannels(outC)
{
    weights.resize(outC, std::vector<float>(inC));
    bias.resize(outC);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5, 0.5);

    for (int i = 0; i < outC; i++) {
        bias[i] = dis(gen);

        for (int j = 0; j < inC; j++) {
            weights[i][j] = dis(gen);
        }
    }
}

std::vector<float> PointwiseConv::forward(
    const std::vector<std::vector<float>>& input)
{
    std::vector<float> output(outChannels, 0);

    // flatten input
    std::vector<float> flat;
    for (auto& row : input)
        for (float v : row)
            flat.push_back(v);

    for (int i = 0; i < outChannels; i++) {
        float sum = bias[i];

        for (int j = 0; j < flat.size(); j++) {
            sum += flat[j] * weights[i][j % inChannels];
        }

        output[i] = sum;
    }

    return output;
}