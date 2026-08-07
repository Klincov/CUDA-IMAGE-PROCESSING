cmake_minimum_required(VERSION 3.18)

# CUDA mora biti naveden kao jezik OVDE, pre project(), da bi CMake
# znao da traži nvcc od samog starta.
project(ImageFilterBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CUDA podešavanja ---
# CUDA_ARCHITECTURES govori nvcc-u za koju generaciju GPU-a da kompajlira.
# 75 = Turing (RTX 20xx), 86 = Ampere (RTX 30xx), 89 = Ada (RTX 40xx).
# Ako nisi siguran koju karticu imaš, "native" (CMake 3.24+) automatski
# detektuje karticu na mašini na kojoj se kompajlira.
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_CUDA_STANDARD 17)

# --- OpenCV ---
find_package(OpenCV REQUIRED)

# --- OpenMP ---
# Ovo NE zahteva nvcc niti poseban jezik - samo dodaje kompajlerski flag
# (-fopenmp) i link-uje libgomp. Zato je mnogo jednostavnije od CUDA dela.
find_package(OpenMP REQUIRED)

# --- Izvorni fajlovi ---
# .cpp i .cu fajlovi se mogu mešati u istoj listi - CMake sam prepoznaje
# ekstenziju i prosledjuje .cu fajlove nvcc-u, a .cpp fajlove g++/clang-u.
file(GLOB_RECURSE SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

add_executable(filter_bench ${SOURCES} "src/app/image_io.cpp" "src/app/image_io.h" "src/filters/sequential/GaussianBlurFilterSeq.h" "src/filters/openmp/GaussianBlurFilterOMP.h")

target_include_directories(filter_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(filter_bench PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

# Omogućava da .cpp fajlovi koji pozivaju CUDA wrapper funkcije
# ispravno linkuju sa CUDA runtime-om (cudart).
set_target_properties(filter_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
