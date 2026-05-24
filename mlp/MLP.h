#pragma once
#include <vector>
#include "core/Model.h"
#include "Layer.h"

class MLP : public Model {
public:
    MLP(const std::vector<int>& topology, double lr = 0.1);

    std::vector<float> forward(const std::vector<float>& input) override;

    void backward(const std::vector<float>& target) override;

    void update(float lr) override;
private:
    std::vector<Layer> layers;
    double learningRate;

    std::vector<float> lastOutput;
};
