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
      grann::cout << "Using L2 Distance Function" << std::endl;
      return new grann::DistanceFastL2<float>();
    } else if (m == grann::Metric::L2) {
      return new grann::DistanceL2();
    } else if (m == grann::Metric::INNER_PRODUCT) {
      grann::cout << "Using Inner Product Function" << std::endl;
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
  ANNIndex<T>::ANNIndex(Metric m, const char *filename,
                        std::vector<_u32> &list_of_tags)
      : _metric(m), _has_built(false) {
    // data is stored to _num_points * aligned_dim matrix with necessary
    // zero-padding
    load_aligned_bin<T>(std::string(filename), _data, _num_points, _dim,
                        _aligned_dim);

    this->_distance = ::get_distance_function<T>(m);
    this->_distance_float = ::get_distance_function<float>(m);

    if (list_of_tags.size() != _num_points && list_of_tags.size() != 0) {
      std::stringstream stream;
      stream << "Mismatch in number of points in data and id_map." << std::endl;
      std::cerr << stream.str() << std::endl;
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
    }
    _tag_map = new _u32[_num_points];
    if (list_of_tags.size() != 0)
      std::memcpy(_tag_map, list_of_tags.data(), _num_points * sizeof(_u32));
    else {
      for (_u32 i = 0; i < _num_points; i++) {
        _tag_map[i] = i;
      }
    }
  }

  template<typename T>
  ANNIndex<T>::ANNIndex(Metric m) : _metric(m), _has_built(false) {
    this->_distance = ::get_distance_function<T>(m);
    this->_distance_float = ::get_distance_function<float>(m);
    _num_points = 0;
  }

  template<typename T>
  ANNIndex<T>::~ANNIndex() {
    delete this->_distance;
    delete this->_distance_float;
    aligned_free(_data);
    delete[] _tag_map;
  }

  template<typename T>
  void ANNIndex<T>::parse_label_file(std::string map_file) {
    //_filtered_ann = 1;
		_filtered_index = true;
    std::ifstream infile(map_file);
    std::string   line, token;
    unsigned      line_cnt = 0;

    while (std::getline(infile, line)) {
      std::istringstream       iss(line);
      std::vector<std::string> lbls(0);
      // long int              val;
      while (getline(iss, token, ',')) {
        token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
        lbls.push_back(token);
        _labels.insert(token);
				_labels_to_pts[token].push_back(line_cnt);
        if (_filter_to_medoid_id.find(token) == _filter_to_medoid_id.end())
          { _filter_to_medoid_id[token] = line_cnt;
        }
      }
      if (lbls.size() <= 0) {
        std::cout << "No label found";
        exit(-1);
      }
      std::sort(lbls.begin(), lbls.end());
      _pts_to_labels.push_back(lbls);
      line_cnt++;
    }
    std::cout << "Identified " << _labels.size() << " distinct label(s)"
              << std::endl;
  }

  template<typename T>
  void ANNIndex<T>::save_data_and_tags(const std::string index_file) {
    std::string data_file = index_file + "_data.bin";
    std::string tag_file = index_file + "_tags.bin";
    grann::save_data_in_original_dimensions<T>(data_file, _data, _num_points,
                                               _aligned_dim, _dim);
    grann::save_bin<_u32>(tag_file, _tag_map, _num_points, 1);
  }

  template<typename T>
  void ANNIndex<T>::load_data_and_tags(const std::string index_file) {
    std::string data_file = index_file + "_data.bin";
    std::string tag_file = index_file + "_tags.bin";
    _u64        num_tags, tmp_dim;
    grann::load_aligned_bin<T>(data_file, _data, _num_points, _dim,
                               _aligned_dim);
    grann::load_bin<_u32>(tag_file, _tag_map, num_tags, tmp_dim);
    if (num_tags != _num_points) {
      grann::cout << "Error! Mismatch between number of tags and number of "
                     "data points. Exitting."
                  << std::endl;
      exit(-1);
    }
  }

  template<typename T>
  _u32 ANNIndex<T>::process_candidates_into_best_candidates_pool(
      const T *&node_coords, std::vector<_u32> &cand_list,
      std::vector<Neighbor> &top_L_candidates, const _u32 maxListSize,
      _u32 &curListSize, tsl::robin_set<_u32> &already_inserted,
      _u32 &total_comparisons) {
    _u32 best_inserted_position = maxListSize;
    for (unsigned m = 0; m < cand_list.size(); ++m) {
      unsigned id = cand_list[m];
      if (_search_filter != "") {
        bool cont = false;
        for (auto z : _pts_to_labels[id]) {
          if (z == _search_filter) {
            cont = true;
            break;
          }
        }
        if (cont)
          continue;
      }
      if (already_inserted.find(id) == already_inserted.end()) {
        already_inserted.insert(id);

        if ((m + 1) < cand_list.size()) {
          auto nextn = cand_list[m + 1];
          grann::prefetch_vector(
              (const char *) this->_data + this->_aligned_dim * (_u64) nextn,
              sizeof(T) * this->_aligned_dim);
        }

        total_comparisons++;
        float dist = this->_distance->compare(
            node_coords, this->_data + this->_aligned_dim * (_u64) id,
            (unsigned) this->_aligned_dim);

        if (curListSize > 0 &&
            dist >= top_L_candidates[curListSize - 1].distance &&
            (curListSize == maxListSize))
          continue;

        Neighbor nn(id, dist, true);
        unsigned r = InsertIntoPool(top_L_candidates.data(), curListSize, nn);
        if (curListSize < maxListSize)
          ++curListSize;  // candidate_list has grown by +1
        if (r < best_inserted_position)
          best_inserted_position = r;
      }
    }
    return best_inserted_position;
  }

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T>
  unsigned ANNIndex<T>::calculate_medoid_of_data() {
    // allocate and init centroid
    float *center = new float[this->_aligned_dim]();
    for (_u64 j = 0; j < this->_aligned_dim; j++)
      center[j] = 0;

    for (_u64 i = 0; i < this->_num_points; i++)
      for (_u64 j = 0; j < this->_aligned_dim; j++)
        center[j] += this->_data[i * this->_aligned_dim + j];

    for (_u64 j = 0; j < this->_aligned_dim; j++)
      center[j] /= this->_num_points;

    // compute all to one distance
    float *distances = new float[this->_num_points]();
#pragma omp parallel for schedule(static, 65536)
    for (_s64 i = 0; i < (_s64) this->_num_points; i++) {
      // extract point and distance reference
      float   &dist = distances[i];
      const T *cur_vec = this->_data + (i * (_u64) this->_aligned_dim);
      dist = 0;
      float diff = 0;
      for (_u64 j = 0; j < this->_aligned_dim; j++) {
        diff = (center[j] - cur_vec[j]) * (center[j] - cur_vec[j]);
        dist += diff;
      }
    }
    // find imin
    unsigned min_idx = 0;
    float    min_dist = distances[0];
    for (unsigned i = 1; i < this->_num_points; i++) {
      if (distances[i] < min_dist) {
        min_idx = i;
        min_dist = distances[i];
      }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
  }

  // EXPORTS
  template class ANNIndex<float>;
  template class ANNIndex<int8_t>;
  template class ANNIndex<uint8_t>;
}  // namespace grann
