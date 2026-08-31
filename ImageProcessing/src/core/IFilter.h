#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class IFilter {
public:
    virtual cv::Mat apply(const cv::Mat& input) = 0;
    virtual std::string name() const = 0;
    virtual ~IFilter() = default;
};
