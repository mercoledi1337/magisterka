#include "PoolLayer.h"
#include <algorithm>

PoolLayer::PoolLayer(int size) : poolSize(size) {}

std::vector<std::vector<float>> PoolLayer::forward(
    const std::vector<std::vector<float>>& input)
{
    int outSize = input.size() / poolSize;

    std::vector<std::vector<float>> output(
        outSize, std::vector<float>(outSize, 0)
    );

    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < outSize; j++) {

            float maxVal = -1e9;

            for (int pi = 0; pi < poolSize; pi++) {
                for (int pj = 0; pj < poolSize; pj++) {
                    maxVal = std::max(
                        maxVal,
                        input[i * poolSize + pi][j * poolSize + pj]
                    );
                }
            }

            output[i][j] = maxVal;
        }
    }

    return output;
}