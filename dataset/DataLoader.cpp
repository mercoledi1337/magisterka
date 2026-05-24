#include "DataLoader.h"

DataLoader::DataLoader(Dataset& d, int batchSize)
{
    this->dataset = &d;
    this->batchSize = batchSize;
}

bool DataLoader::hasNext() const
{
    return index < dataset->data.size();
}

Batch DataLoader::next()
{
    Batch batch;

    for (int i = 0;
         i < batchSize && index < dataset->data.size();
         i++)
    {
        batch.x.push_back(dataset->data[index].x);
        batch.y.push_back(dataset->data[index].y);
        index++;
    }

    return batch;
}

void DataLoader::reset()
{
    index = 0;
}