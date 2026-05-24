#include "ConvLayer.h"
#include <random>

ConvLayer::ConvLayer(int k, int s, int inputW)
    : kernelSize(k), stride(s), inputWidth(inputW)
{
    kernel.resize(k, std::vector<float>(k));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5, 0.5);

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            kernel[i][j] = dis(gen);

    bias = dis(gen);
}

float ConvLayer::relu(float x) {
    return x > 0 ? x : 0;
}

std::vector<std::vector<float>> ConvLayer::forward(
    const std::vector<std::vector<float>>& input)
{
    int outSize = inputWidth - kernelSize + 1;

    output.assign(outSize, std::vector<float>(outSize, 0));

    for (int i = 0; i < outSize; i += stride) {
        for (int j = 0; j < outSize; j += stride) {

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