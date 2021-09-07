// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once


#include "utils.h"
//#include "distance.h"

namespace grann {
  template<typename T>
  class Dataset {
   public:
    Dataset(Metric m, const char *filename);
    ~Dataset();

    void save(const char *filename);
    void load(const char *filename);

    /*  Internals of the library */
   private:
    Metric       _metric = grann::L2;
    size_t       _dim;
    size_t       _aligned_dim; // for faster distance computations using AVX instructions, we round up dimension to multiple of 8
    T *          _data;
    size_t       _num_points;  // number of points in the dataset
//    Distance<T> *_distance;
  };
}  // namespace grann
