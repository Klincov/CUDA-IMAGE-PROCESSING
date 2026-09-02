#pragma once
#include <omp.h>

void DisplayMaxThreads() {
	int maxThreads = omp_get_max_threads();
	std::cout << "Max CPU threads available: " << maxThreads << std::endl;
};