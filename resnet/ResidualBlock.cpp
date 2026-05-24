#include "ResidualBlock.h"
#include <random>

ResidualBlock::ResidualBlock(int size) : size(size)
{
    W1.resize(size, std::vector<float>(size));
    W2.resize(size, std::vector<float>(size));

    b1.resize(size);
    b2.resize(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5, 0.5);

    for (int i = 0; i < size; i++) {
        b1[i] = dis(gen);
        b2[i] = dis(gen);

        for (int j = 0; j < size; j++) {
            W1[i][j] = dis(gen);
            W2[i][j] = dis(gen);
        }
    }
}

float ResidualBlock::relu(float x) {
    return x > 0 ? x : 0;
}

std::vector<std::vector<float>> ResidualBlock::forward(
    const std::vector<std::vector<float>>& x)
{
    int n = x.size();

    std::vector<std::vector<float>> out(n,
        std::vector<float>(n, 0));

    // F(x)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            float sum = b1[i];

            for (int k = 0; k < n; k++) {
                sum += x[i][k] * W1[k][j];
            }

            out[i][j] = relu(sum);
        }
    }

    // drugi layer
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            float sum = b2[i];

            for (int k = 0; k < n; k++) {
                sum += out[i][k] * W2[k][j];
            }

            out[i][j] = relu(sum);
        }
    }

    // 🔥 SKIP CONNECTION: F(x) + x
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out[i][j] += x[i][j];
        }
    }

    return out;
}