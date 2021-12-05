#include <chrono>
#include <random>
#include "utils.h"
#include <omp.h> // TODO: parallelise the code

namespace grann {
    void random_gaussian(const char* filename, _u32 num_vectors, _u32 dim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> normal(0, 1);

        float* data = new float[num_vectors * dim];
        
#pragma omp parallel for schedule(static, 64)
        for(_u32 ii = 0; ii < num_vectors; ++ii) {
            for(_u32 jj = 0; jj < dim; ++jj) {
                data[(ii * dim) + jj] = normal(gen);
            }
        }

        grann::save_bin<float>(filename, data, num_vectors, dim);
        delete[] data;
        return;
    }

    void random_cube(const char* filename, _u32 num_vectors, _u32 dim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> cube(-1.0, 1.0);

        float* data = new float[num_vectors * dim];

#pragma omp parallel for schedule(static, 64)
        for(_u32 ii = 0; ii < num_vectors; ++ii) {
            for(_u32 jj = 0; jj < dim; ++jj) {
                data[(ii * dim) + jj] = cube(gen);
            }
        }

        grann::save_bin<float>(filename, data, num_vectors, dim);
        delete[] data;
        return;
    }
}