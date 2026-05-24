#ifndef POOL_LAYER_H
#define POOL_LAYER_H

#include <vector>

class PoolLayer {
public:
    int poolSize;

    PoolLayer(int size);

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& input
    );
};

#endif