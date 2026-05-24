#include "MobileNet.h"

MobileNet::MobileNet()
    : dw1(3),
      pw1(64, 32)
{}

std::vector<float> MobileNet::forward(const std::vector<float>& x)
{
    // reshape 64x64
    std::vector<std::vector<float>> img(64,
        std::vector<float>(64));

    int idx = 0;
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
            img[i][j] = x[idx++];

    // 🔥 depthwise conv
    auto dw = dw1.forward(img);

    // 🔥 pointwise conv
    auto pw = pw1.forward(dw);

    // classifier head
    output.resize(3);

    for (int i = 0; i < 3; i++) {
        output[i] = pw[i % pw.size()];
    }

    return output;
}