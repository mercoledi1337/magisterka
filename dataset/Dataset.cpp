#include "Dataset.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>

void Dataset::load(const std::string& path, int numClasses)
{
    for (int c = 0; c < numClasses; c++)
    {
        std::string classPath =
            path + "/class_" + std::to_string(c);

        for (auto& file :
             std::filesystem::directory_iterator(classPath))
        {
            cv::Mat img =
                cv::imread(file.path(), cv::IMREAD_GRAYSCALE);

            if (img.empty()) continue;

            cv::resize(img, img, cv::Size(64, 64));

            Sample s;

            s.x.reserve(4096);

            for (int i = 0; i < 64; i++)
                for (int j = 0; j < 64; j++)
                    s.x.push_back(img.at<uchar>(i,j) / 255.0f);

            s.y.assign(numClasses, 0.0f);
            s.y[c] = 1.0f;

            data.push_back(s);
        }
    }
}