#include "MLP.h"
#include <ostream>


MLP::MLP(const std::vector<int> &topology, const double lr) : learningRate(lr) {
    for (size_t i = 1; i < topology.size(); ++i) {
        layers.push_back(Layer(topology[i], topology[i - 1]));
    }
}

std::vector<float> MLP::forward(const std::vector<float>& input)
{
    std::vector<float> current = input;

    for (auto& layer : layers)
    {
        layer.forward(current);
        current = layer.output;
    }

    lastOutput = current;
    return current;
}

void MLP::backward(const std::vector<float>& target)
{
    int L = layers.size();

    // output layer error
    for (int i = 0; i < layers[L - 1].output.size(); i++)
    {
        float out = layers[L - 1].output[i];
        layers[L - 1].deltas[i] =
            (out - target[i]) * Layer::sigmoidDerivative(out);
    }

    // hidden layers
    for (int l = L - 2; l >= 0; l--)
    {
        for (int i = 0; i < layers[l].output.size(); i++)
        {
            float error = 0.0f;

            for (int j = 0; j < layers[l + 1].output.size(); j++)
            {
                error += layers[l + 1].weights[j][i] *
                         layers[l + 1].deltas[j];
            }

            float out = layers[l].output[i];
            layers[l].deltas[i] =
                error * Layer::sigmoidDerivative(out);
        }
    }
}
void MLP::update(float lr)
{
    for (int l = 0; l < layers.size(); l++)
    {
        auto& layer = layers[l];

        std::vector<float> input =
            (l == 0) ? layer.lastInput : layers[l - 1].output;

        for (int i = 0; i < layer.weights.size(); i++)
        {
            for (int j = 0; j < layer.weights[i].size(); j++)
            {
                layer.weights[i][j] -=
                    lr * layer.deltas[i] * input[j];
            }

            layer.biases[i] -= lr * layer.deltas[i];
        }
    }
}



