#pragma once
#include "CPUTiming.h"

class ICPUTimed {
public:
    virtual CpuTimingResult applyTimed(const cv::Mat& input, cv::Mat& output, int numThreads = 0) = 0;
    virtual ~ICPUTimed() = default;
};

