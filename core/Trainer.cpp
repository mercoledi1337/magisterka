#include "Trainer.h"
#include "Loss.h"
#include <iostream>

void Trainer::train(Model& model, Dataset& data)
{
    for (auto& sample : data.data)
    {
        auto pred = model.forward(sample.x);
        model.backward(sample.y);
        model.update(0.01f);
    }
}