#include "Loss.h"
#include <cmath>

float Loss::crossEntropy(
    const std::vector<float>& pred,
    const std::vector<float>& target)
{
    float loss = 0.0f;

    for (int i = 0; i < pred.size(); i++) {
        float p = std::max(pred[i], 1e-7f);
        loss += -target[i] * log(p);
    }

    return loss;
}

std::vector<float> Loss::crossEntropyGrad(
    const std::vector<float>& pred,
    const std::vector<float>& target)
{
    std::vector<float> grad(pred.size());

    for (int i = 0; i < pred.size(); i++) {
        grad[i] = pred[i] - target[i];
    }

    return grad;
}