#ifndef MODEL_H
#define MODEL_H

#include <vector>

class Model {
public:
    virtual std::vector<float> forward(const std::vector<float>& x) = 0;

    virtual void backward(
        const std::vector<std::vector<float>>& x,
        const std::vector<std::vector<float>>& y,
        float lr
    ) = 0;

    virtual ~Model() {}
};

#endif