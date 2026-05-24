#include "MLP.h"
#include <fstream>
#include <iostream>
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
void MLP::train(const std::vector<std::vector<float>>& inputs,
                      const std::vector<std::vector<float>>& targets,
                      const int epochs) {
    for (int e = 0; e < epochs; ++e) {
        float totalError = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<float> prediction = feedForward(inputs[i]);

            for (size_t j = 0; j < targets[i].size(); ++j) {
                totalError += std::pow(targets[i][j] - prediction[j], 2);
            }

            backpropagate(targets[i]);
        }
        if (e % 100 == 0) {
            std::cout << "Epoch: " << e << "/ mean erros" << totalError / inputs.size() << std::endl;
        }
    }
}

void MLP::saveWeights(std::string filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "Can't open file" << std::endl;
        return;
    }

    for (auto &layer : layers) {
        for (const auto &neuronWeights : layer.weights) {
            for (float w : neuronWeights) {
                file << w << " ";
            }
            file << "\n";
        }
        for (float b : layer.biases) {
            file << b << " ";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Saved weights to " << filename << std::endl;
};

void MLP::loadWeights(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Blad: Nie mozna otworzyc pliku " << filename << " do odczytu!" << std::endl;
        return;
    }

    for (auto &layer : layers) {

        for (auto &neuronWeights : layer.weights) {
            for (float &w : neuronWeights) {
                if (!(file >> w)) break;
            }
        }

        for (float &b : layer.biases) {
            if (!(file >> b)) break;
        }
    }

    file.close();
    std::cout << "--- Wagi wczytane pomyslnie z pliku: " << filename << " ---" << std::endl;
}