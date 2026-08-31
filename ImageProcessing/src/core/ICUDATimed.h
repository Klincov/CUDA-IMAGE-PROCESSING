#pragma once
#include "CUDATiming.cuh"
#include <string>

class ICUDATimed {
public:
    virtual CudaTimingResult applyTimed(const cv::Mat& input, cv::Mat& output) = 0;
    virtual ~ICUDATimed() = default;
};
