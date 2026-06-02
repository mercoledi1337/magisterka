#include "dataset/DataLoader.h"

#include "core/Trainer.h"
#include "core/Evaluator.h"

#include "mlp/MLP.h"
// później możesz podmienić na:
// #include "cnn/CNN.h"
// #include "resnet/ResNet.h"

#include <iostream>
#include <vector>

int main()
{
    // =====================================
    // 1. LOAD DATA
    // =====================================

    Dataset trainDataset;
    trainDataset.load("data/train", 3);
    trainDataset.shuffle();

    Dataset testDataset;
    testDataset.load("data/test", 3);

    Dataset valDataset;
    valDataset.load("data/val", 3);

    std::cout << "Datasets loaded\n";

    // =====================================
    // 2. CREATE DATALOADER
    // =====================================

    int batchSize = 16;

    DataLoader trainLoader(trainDataset, batchSize);

    // =====================================
    // 3. CREATE MODEL
    // =====================================
    //
    // 64x64 grayscale:
    // 64 * 64 = 4096
    //
    // output = 3 classes
    //

    std::vector<int> topology =
    {
        4096,
        128,
        64,
        3
    };

    MLP model(topology, 0.01);

    // =====================================
    // 4. TRAINING
    // =====================================

    Trainer trainer;

    int epochs = 10;
    float learningRate = 0.01f;

    trainer.train(
        model,
        trainLoader,
        epochs,
        learningRate
    );

    // =====================================
    // 5. EVALUATION
    // =====================================

    Evaluator evaluator;

    float testAccuracy =
        evaluator.accuracy(model, testDataset);

    float valAccuracy =
        evaluator.accuracy(model, valDataset);

    // =====================================
    // 6. RESULTS
    // =====================================

    std::cout << "\n===== RESULTS =====\n";

    std::cout
        << "Test Accuracy: "
        << testAccuracy * 100.0f
        << "%\n";

    std::cout
        << "Validation Accuracy: "
        << valAccuracy * 100.0f
        << "%\n";

    return 0;
}