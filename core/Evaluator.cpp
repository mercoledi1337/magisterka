#include "Evaluator.h"

int argmax(const std::vector<float>& v) {
    int idx = 0;
    float best = v[0];

    for (int i = 1; i < v.size(); i++) {
        if (v[i] > best) {
            best = v[i];
            idx = i;
        }
    }

    return idx;
}

float Evaluator::accuracy(Model& model,
                           const std::vector<std::vector<float>>& x,
                           const std::vector<std::vector<float>>& y)
{
    int correct = 0;

    for (int i = 0; i < x.size(); i++) {
        auto pred = model.forward(x[i]);

        if (argmax(pred) == argmax(y[i]))
            correct++;
    }

    return (float)correct / x.size();
}