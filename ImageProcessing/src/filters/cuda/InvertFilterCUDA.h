#pragma once
#include "core/IFilter.h"
#include "invert_kernel.cuh"

// Ovo je "obican" C++ header - moze da se ukljuci bilo gde u projektu
// (main.cpp, meni, benchmark) bez ikakve CUDA zavisnosti u samom kodu.
// Sva CUDA magija je sakrivena iza launchInvertKernel().
class InvertFilterCUDA : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input) override {
        // cv::Mat mora biti kontinualan u memoriji (isContinuous()) da bi
        // ovakav direktan pristup .data pokazivacu bio ispravan. OpenCV
        // slike ucitane preko imread/clone obicno jesu, ali je dobra
        // navika da se to eksplicitno proveri/obezbedi.
        cv::Mat output = input.clone();
        if (!output.isContinuous()) {
            output = output.clone();
        }

        launchInvertKernel(output.data, output.cols, output.rows, output.channels());
        return output;
    }

    std::string name() const override { return "Invert (CUDA)"; }
};
