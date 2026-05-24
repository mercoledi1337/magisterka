#include "Flatten.h"

std::vector<float> Flatten::forward(
    const std::vector<std::vector<float>>& input)
{
    std::vector<float> out;

    for (const auto& row : input)
        for (float v : row)
            out.push_back(v);

    return out;
}