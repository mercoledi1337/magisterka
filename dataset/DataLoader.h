#pragma once
#include "Dataset.h"
#include "Batch.h"

class DataLoader {
private:
    Dataset* dataset;
    size_t index = 0;
    int batchSize;

public:
    DataLoader(Dataset& d, int batchSize);

    bool hasNext() const;

    Batch next();
    void reset();
};