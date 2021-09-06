// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once


#include "utils.h"
#include "distance.h"

namespace grann {
  template<typename T>
  class Dataset {
   public:
    GRANN_DLLEXPORT Dataset(Metric m, const char *filename);
    GRANN_DLLEXPORT ~Dataset();

    GRANN_DLLEXPORT void save(const char *filename);
    GRANN_DLLEXPORT void load(const char *filename);

    /*  Internals of the library */
   private:
    Metric       _metric = grann::L2;
    size_t       _dim;
    size_t       _aligned_dim;
    T *          _data;
    size_t       _nd;  // number of active points i.e. existing in the graph
    Distance<T> *_distance;
  };
}  // namespace grann
