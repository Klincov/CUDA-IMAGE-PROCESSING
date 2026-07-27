#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// Zajednički interfejs koji svaka implementacija (sekvencijalna, OpenMP,
// CUDA) mora da poštuje. Benchmark harness i meni rade samo protiv
// ovog interfejsa - nikad ne znaju da li je pozvana CPU ili GPU verzija.
class IFilter {
public:
    virtual cv::Mat apply(const cv::Mat& input) = 0;
    virtual std::string name() const = 0;
    virtual ~IFilter() = default;
};
