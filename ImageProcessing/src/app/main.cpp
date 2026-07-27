#include <iostream>
#include <memory>
#include <vector>
#include "core/IFilter.h"
#include "filters/sequential/InvertFilterSeq.h"
#include "filters/openmp/InvertFilterOMP.h"
#include "filters/cuda/InvertFilterCUDA.h"

// Ovo NIJE finalni meni - samo dokaz da se ceo pipeline (C++, OpenMP i
// CUDA u istom executable-u) kompajlira, linkuje i daje ispravan
// (isti) rezultat pre nego sto se krene u pisanje pravih filtera.
int main() {
    // 100x100 crvena slika, dovoljno mala da se rezultat lako proveri okom.
    cv::Mat testImage(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));

    std::vector<std::unique_ptr<IFilter>> filters;
    filters.push_back(std::make_unique<InvertFilterSeq>());
    filters.push_back(std::make_unique<InvertFilterOMP>());
    filters.push_back(std::make_unique<InvertFilterCUDA>());

    for (auto& filter : filters) {
        cv::Mat result = filter->apply(testImage);
        cv::Vec3b samplePixel = result.at<cv::Vec3b>(0, 0);

        std::cout << filter->name() << " -> pixel(0,0) = ["
                  << (int)samplePixel[0] << ", "
                  << (int)samplePixel[1] << ", "
                  << (int)samplePixel[2] << "]"
                  << " (ocekivano: [255, 255, 0])" << std::endl;
    }

    return 0;
}
