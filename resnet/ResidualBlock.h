#include "CNN.h"

CNN::CNN()
    : conv1(3, 1, 64),
      pool1(2)
{
    fcWeights.resize(1000);
}

std::vector<float> CNN::forward(const std::vector<float>& x)
{
    // 1. reshape 1D → 2D (64x64)
    std::vector<std::vector<float>> img(64,
        std::vector<float>(64));

    int idx = 0;
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
            img[i][j] = x[idx++];

    // 2. conv
    auto c1 = conv1.forward(img);

    // 3. pooling
    auto p1 = pool1.forward(c1);

    // 4. flatten
    auto f = flatten.forward(p1);

    // 5. FC (prosty output layer)
    output.resize(3, 0);

    for (int i = 0; i < 3; i++) {
        float sum = 0;
        for (int j = 0; j < f.size(); j++) {
            sum += f[j] * fcWeights[j % fcWeights.size()];
        }
        output[i] = sum;
    }

    return output;
}