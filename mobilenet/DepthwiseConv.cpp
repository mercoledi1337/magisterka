#include "DepthwiseConv.h"
#include <random>

DepthwiseConv::DepthwiseConv(int k)
    : kernelSize(k)
{
    kernel.resize(k, std::vector<float>(k));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5, 0.5);

    bias = dis(gen);

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            kernel[i][j] = dis(gen);
}

float DepthwiseConv::relu(float x) {
    return x > 0 ? x : 0;
}

std::vector<std::vector<float>> DepthwiseConv::forward(
    const std::vector<std::vector<float>>& input)
{
    int n = input.size();
    int outSize = n - kernelSize + 1;

    std::vector<std::vector<float>> output(
        outSize, std::vector<float>(outSize, 0)
    );

    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < outSize; j++) {

            float sum = bias;

            for (int ki = 0; ki < kernelSize; ki++) {
                for (int kj = 0; kj < kernelSize; kj++) {
                    sum += input[i + ki][j + kj] *
                           kernel[ki][kj];
                }
            }

            output[i][j] = relu(sum);
        }
    }

    return output;
}