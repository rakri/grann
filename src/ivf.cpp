// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "ivf.h"
#include "partition_and_pq.h"
#include "math_utils.h"
#include <iomanip>

namespace grann {

  // Initialize a generic graph-based index with metric m, load the data of type
  // T with filename (bin)
  template<typename T>
  IVFIndex<T>::IVFIndex(Metric m, const char *filename,
                        std::vector<_u32> &list_of_tags)
      : ANNIndex<T>(m, filename,
                    list_of_tags) {  // Graph Index class constructor loads the
                                     // data and sets num_points, dim, etc.
    _num_clusters = 0;
  }

  template<typename T>
  IVFIndex<T>::IVFIndex(Metric m)
      : ANNIndex<T>(m) {  // Graph Index class constructor empty for load.
    _num_clusters = 0;
  }

  template<typename T>
  void IVFIndex<T>::save(const char *filename) {
    ANNIndex<T>::save_data_and_tags(filename);
    std::string centers_file(filename);
    std::string index_file(filename);
    centers_file += "_centers.bin";
    grann::save_bin<float>(centers_file, _cluster_centers, _num_clusters,
                           this->_aligned_dim);

    std::ofstream out(index_file, std::ios::binary | std::ios::out);
    _u64          total_count = 0;

    for (unsigned i = 0; i < this->_num_clusters; i++) {
      unsigned GK = (unsigned) _inverted_index[i].size();
      grann::cout << GK << std::endl;
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _inverted_index[i].data(), GK * sizeof(unsigned));
      total_count += GK;
    }
    out.close();

    grann::cout << "Written a total of " << total_count
                << " points to inverted index file." << std::endl;
  }

  template<typename T>
  void IVFIndex<T>::load(const char *filename) {
    ANNIndex<T>::load_data_and_tags(filename);
    std::string centers_file(filename);
    std::string index_file(filename);
    centers_file += "_centers.bin";
    _u64 tmp_dim1, tmp_dim2;  // should be same as aligned_dim
    grann::load_aligned_bin<float>(centers_file, _cluster_centers,
                                   _num_clusters, tmp_dim1, tmp_dim2);

    std::ifstream in(index_file, std::ios::binary | std::ios::out);
    _u64          total_count = 0;

    _inverted_index.resize(_num_clusters);
    for (unsigned i = 0; i < this->_num_clusters; i++) {
      unsigned GK;
      in.read((char *) &GK, sizeof(unsigned));
      grann::cout << GK << std::endl;
      _inverted_index[i].resize(GK);
      in.read((char *) _inverted_index[i].data(), GK * sizeof(unsigned));
      total_count += GK;
    }
    in.close();
    grann::cout << "Read a total of " << total_count
                << " points from inverted index file." << std::endl;
  }

  template<typename T>
  void IVFIndex<T>::build(Parameters &build_params) {
    _num_clusters = build_params.Get<_u32>("num_clusters");
    float training_prob = build_params.Get<float>("training_rate");

    _inverted_index.resize(_num_clusters);
    _cluster_centers = new float[_num_clusters * this->_aligned_dim];

    float *train_data_float;
    _u64   num_train;

    gen_random_slice<T>(this->_data, this->_num_points, this->_aligned_dim,
                        training_prob, train_data_float, num_train);
    grann::cout << "Going to train the cluster centers over " << num_train
                << " points " << std::endl;

    math_utils::kmeans_plus_plus_centers(train_data_float, num_train,
                                         this->_aligned_dim, _cluster_centers,
                                         _num_clusters);

    math_utils::run_lloyds(train_data_float, num_train, this->_aligned_dim,
                           _cluster_centers, _num_clusters, MAX_K_MEANS_REPS,
                           nullptr, nullptr);
    grann::cout << "Done. Now going to build the index." << std::endl;

    std::vector<_u32> closest_centers(this->_num_points);
    float *data_float = new float[this->_num_points * this->_aligned_dim];

    grann::convert_types(this->_data, data_float, this->_num_points,
                         this->_aligned_dim);
    math_utils::compute_closest_centers(
        data_float, this->_num_points, this->_aligned_dim, _cluster_centers,
        _num_clusters, 1, closest_centers.data(), _inverted_index.data(),
        nullptr);

    delete[] train_data_float;
    delete[] data_float;
  }

  // returns # results found (will be <= res_count)
  template<typename T>
  _u32 IVFIndex<T>::search(const T *query, _u32 res_count,
                           Parameters &search_params, _u32 *indices,
                           float *distances, QueryStats *stats) {
  }

  // EXPORTS
  template class IVFIndex<float>;
  template class IVFIndex<int8_t>;
  template class IVFIndex<uint8_t>;
}  // namespace grann
