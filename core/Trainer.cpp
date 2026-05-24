#include "Trainer.h"
#include "Loss.h"
#include <iostream>

void Trainer::train(Model& model, Dataset& data)
{
    for each epoch:
    for each sample:
        model.forward()
        model.backward()
        model.update()
}