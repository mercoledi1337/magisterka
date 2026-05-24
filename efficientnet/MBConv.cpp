#include "MBConv.h"
#include <random>

MBConv::MBConv(int c)
    : channels(c), se(c)
{
    dwKernel.resize(3, std::vector<float>(3));
    pwWeights.resize(c, std::vector<float>(c));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5, 0.5);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            dwKernel[i][j] = dis(gen);

    for (int i = 0; i < c; i++)
        for (int j = 0; j < c; j++)
            pwWeights[i][j] = dis(gen);
}

float MBConv::relu(float x) {
    return x > 0 ? x : 0;
}

std::vector<float> MBConv::forward(
    const std::vector<std::vector<float>>& input)
{
    int n = input.size();

    // depthwise conv (simplified)
    std::vector<float> dw(n * n);

    int idx = 0;

    for (int i = 0; i < n - 2; i++) {
        for (int j = 0; j < n - 2; j++) {

            float sum = 0;

            for (int ki = 0; ki < 3; ki++) {
                for (int kj = 0; kj < 3; kj++) {
                    sum += input[i + ki][j + kj] *
                           dwKernel[ki][kj];
                }
            }

            dw[idx++] = relu(sum);
        }
    }

    // SE block
    dw = se.forward(dw);

    // pointwise conv
    std::vector<float> out(channels, 0);

    for (int i = 0; i < channels; i++) {
        float sum = 0;

        for (int j = 0; j < dw.size(); j++) {
            sum += dw[j] * pwWeights[i][j % channels];
        }

        out[i] = relu(sum);
    }

    return out;
}