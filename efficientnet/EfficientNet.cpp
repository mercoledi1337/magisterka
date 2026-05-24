#include "EfficientNet.h"

EfficientNet::EfficientNet(int c)
    : block1(c), block2(c)
{}

std::vector<float> EfficientNet::forward(const std::vector<float>& x)
{
    int n = 64;

    // reshape
    std::vector<std::vector<float>> img(n,
        std::vector<float>(n));

    int idx = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            img[i][j] = x[idx++];

    // MBConv stack
    auto h1 = block1.forward(img);
    auto h2 = block2.forward(img);

    // classification head
    output.resize(3);

    for (int i = 0; i < 3; i++) {
        output[i] =
            h1[i % h1.size()] +
            h2[i % h2.size()];
    }

    return output;
}