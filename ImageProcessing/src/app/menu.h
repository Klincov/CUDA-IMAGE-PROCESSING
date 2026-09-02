#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "core/IFilter.h"

void printMenu();

void printFilterMenu();

// Glavna petlja menija za izbor i primenu filtera nad ucitanom slikom.
void filterMenu(cv::Mat& image);

// Nudi izbor implementacije (Sequential/OMP/CUDA/Validation) za dati skup
// filtera i primenjuje izabranu, ili pokrece validaciju sve tri verzije.
void implementationChoiceMenu(const std::vector<std::unique_ptr<IFilter>>& filters, cv::Mat& image);