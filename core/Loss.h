#ifndef LOSS_H
#define LOSS_H

#include <vector>

class Loss {
public:
    static float crossEntropy(
        const std::vector<float>& pred,
        const std::vector<float>& target
    );

    static std::vector<float> crossEntropyGrad(
        const std::vector<float>& pred,
        const std::vector<float>& target
    );
};

#endif