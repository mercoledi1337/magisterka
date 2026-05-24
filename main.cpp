#include "dataset/Dataset.h"
#include "dataset/DataLoader.h"

#include "mlp/MLP.h"
// później: cnn/CNN.h itd.

#include "core/Evaluator.h"

#include <iostream>

int main()
{
    // 1. LOAD DATA
    Dataset train;
    train.load("data/train", 3);
    train.shuffle();

    Dataset test;
    test.load("data/test", 3);

    // 2. MODEL INIT
    std::vector<int> topology = {4096, 64, 3};
    MLP model(topology, 0.01);

    // 3. TRAINING
    DataLoader loader(train, 16);

    for (int epoch = 0; epoch < 10; epoch++)
    {
        loader.reset();

        while (loader.hasNext())
        {
            Batch batch = loader.next();

            for (int i = 0; i < batch.x.size(); i++)
            {
                model.forward(batch.x[i]);
                model.backward(batch.y[i]);
            }
        }

        std::cout << "Epoch " << epoch << " done\n";
    }

    // 4. EVALUATION
    Evaluator eval;

    float acc = eval.accuracy(model, test.data);

    std::cout << "Test accuracy: " << acc << std::endl;

    return 0;
}