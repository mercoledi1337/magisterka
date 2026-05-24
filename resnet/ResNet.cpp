#include "ResNet.h"

ResNet::ResNet(int size)
    : block1(size),
      block2(size)
{}

std::vector<float> ResNet::forward(const std::vector<float>& x)
{
    int n = 64;

    // reshape 1D → 2D
    std::vector<std::vector<float>> img(n,
        std::vector<float>(n));

    int idx = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            img[i][j] = x[idx++];

    // 🔥 Residual blocks
    auto h1 = block1.forward(img);
    auto h2 = block2.forward(h1);

    // flatten
    std::vector<float> flat;
    for (auto& row : h2)
        for (float v : row)
            flat.push_back(v);

    // simple classifier head
    output.resize(3);

    for (int i = 0; i < 3; i++) {
        float sum = 0;
        for (int j = 0; j < flat.size(); j++) {
            sum += flat[j] * 0.001f; // placeholder weights
        }
        output[i] = sum;
    }

    return output;
}