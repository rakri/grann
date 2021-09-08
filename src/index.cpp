// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "vamana.h"

#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

// only L2 implemented. Need to implement inner product search
namespace {
  template<typename T>
  grann::Distance<T> *get_distance_function(grann::Metric m);

  template<>
  grann::Distance<float> *get_distance_function(grann::Metric m) {
    if (m == grann::Metric::FAST_L2) {
      std::cout << "Using L2 Distance Function" << std::endl;
      return new grann::DistanceFastL2<float>();
    } else if (m == grann::Metric::L2) {
        return new grann::DistanceL2(); 
     } else if (m == grann::Metric::INNER_PRODUCT) {
      std::cout << "Using Inner Product Function" << std::endl;
      return new grann::DistanceInnerProduct<float>();
    } else {
      std::stringstream stream;
      stream << "Only L2/Inner Product metric supported as of now. Email "
                "gopalsr@microsoft.com for anything else."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  grann::Distance<int8_t> *get_distance_function(grann::Metric m) {
    if (m == grann::Metric::L2) {
        return new grann::DistanceL2Int8();
    } else {
      std::stringstream stream;
      stream << "Only L2 metric supported as of now. Email "
                "gopalsr@microsoft.com for anything else."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  grann::Distance<uint8_t> *get_distance_function(grann::Metric m) {
    if (m == grann::Metric::L2) {
      return new grann::DistanceL2UInt8();
    } else {
      std::stringstream stream;
      stream << "Only L2 metric supported as of now. Email "
                "gopalsr@microsoft.com for anything else."
             << std::endl;
      std::cerr << stream.str() << std::endl;
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }
}  // namespace

namespace grann {

  // Initialize an vamana with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T>
  ANNIndex<T>::ANNIndex(Metric m, const char *filename, std::vector<_u32> &list_of_ids)
      : _metric(m), _has_built(false) {

    // data is stored to _num_points * aligned_dim matrix with necessary
    // zero-padding
    load_aligned_bin<T>(std::string(filename), _data, _num_points, _dim, _aligned_dim);

    this->_distance = ::get_distance_function<T>(m);
  }

  template<typename T>
  ANNIndex<T>::~ANNIndex() {
    delete this->_distance;
    aligned_free(_data);
  }


  // EXPORTS
  template GRANN_DLLEXPORT class ANNIndex<float>;
  template GRANN_DLLEXPORT class ANNIndex<int8_t>;
  template GRANN_DLLEXPORT class ANNIndex<uint8_t>;
}  // namespace grann
