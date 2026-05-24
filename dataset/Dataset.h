#pragma once
#include <vector>
#include <string>

struct Sample {
    std::vector<float> x;
    std::vector<float> y;
};

class Dataset {
public:
    std::vector<Sample> data;

    void load(const std::string& path, int numClasses);
    void shuffle();
};