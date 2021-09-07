// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "vamana.h"

#ifdef _WINDOWS
#include <xmmintrin.h>
#endif


namespace grann {

  // Initialize an vamana with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T>
  Dataset<T>::Dataset(Metric m, const char *filename)
      : _metric(m) {
    load_aligned_bin<T>(std::string(filename), _data, _num_points, _dim, _aligned_dim);
    }
 
  template<typename T>
  Dataset<T>::~Dataset() {
    aligned_free(_data);
  }

  // save the data file
  template<typename T>
  void Dataset<T>::save(const char *filename) {
  }

  // load the data (if necessary)
  template<typename T>
  void Dataset<T>::load(const char *filename) {
  }


  // EXPORTS
  template GRANN_DLLEXPORT class Dataset<float>;
  template GRANN_DLLEXPORT class Dataset<int8_t>;
  template GRANN_DLLEXPORT class Dataset<uint8_t>;
}  // namespace grann
