#include "Trainer.h"
#include "Loss.h"
#include <iostream>

void Trainer::train(Model& model,
                    const std::vector<Batch>& data,
                    int epochs,
                    float lr)
{
    for (int e = 0; e < epochs; e++) {

        float totalLoss = 0.0f;

        for (const auto& batch : data) {

            std::vector<std::vector<float>> preds;

            for (const auto& x : batch.x) {
                preds.push_back(model.forward(x));
            }

            for (int i = 0; i < preds.size(); i++) {
                totalLoss += Loss::crossEntropy(preds[i], batch.y[i]);
            }

            model.backward(batch.x, batch.y, lr);
        }

        std::cout << "Epoch " << e
                  << " loss=" << totalLoss << std::endl;
    }
}