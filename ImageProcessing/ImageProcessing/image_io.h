#pragma once
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat loadImage(const std::string& path, bool grayscale = false);
void saveImage(const std::string& path, const cv::Mat& image);
void printImageInfo(const cv::Mat& image, const std::string& label = "");