#include "SEBlock.h"
#include <random>

SEBlock::SEBlock(int c) : channels(c)
{
    weights1.resize(c);
    weights2.resize(c);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < c; i++) {
        weights1[i] = dis(gen);
        weights2[i] = dis(gen);
    }
}

std::vector<float> SEBlock::forward(
    const std::vector<float>& input)
{
    // squeeze (global avg pooling)
    float avg = 0;
    for (float v : input) avg += v;
    avg /= input.size();

    // excitation
    std::vector<float> out(input.size());

    for (int i = 0; i < input.size(); i++) {
        float scale = avg * weights1[i];
        scale = scale * weights2[i];

        out[i] = input[i] * scale;
    }

    return out;
}