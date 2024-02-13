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
  IVFIndex<T>::IVFIndex(Metric m, const char *filename, std::vector<_u32> &list_of_tags, std::string labels_fname)
      : ANNIndex<T>(m, filename,
                    list_of_tags, labels_fname) {  // Graph Index class constructor loads the
                                     // data and sets num_points, dim, etc.
    _num_clusters = 0;
  }

  template<typename T>
  IVFIndex<T>::IVFIndex(Metric m)
      : ANNIndex<T>(m) {  // Graph Index class constructor empty for load.
    _num_clusters = 0;
  }

  template<typename T>
  IVFIndex<T>::~IVFIndex() {
    if (_cluster_centers != nullptr)
      delete[] _cluster_centers;
  }

  template<typename T>
  void IVFIndex<T>::save(const char *filename) {
    ANNIndex<T>::save_data_and_tags_and_labels(filename);
    std::string centers_file(filename);
    std::string index_file(filename);
    centers_file += "_centers.bin";
    grann::save_bin<float>(centers_file, _cluster_centers, _num_clusters,
                           this->_aligned_dim);

    std::ofstream out(index_file, std::ios::binary | std::ios::out);
    _u64          total_count = 0;

    for (unsigned i = 0; i < this->_num_clusters; i++) {
      unsigned GK = (unsigned) _inverted_index[i].size();

      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _inverted_index[i].data(), GK * sizeof(unsigned));
      total_count += GK;
    }
    out.close();

    std::cout << "Written a total of " << total_count
                << " points to inverted index file." << std::endl;
  }

  template<typename T>
  void IVFIndex<T>::load(const char *filename) {
    ANNIndex<T>::load_data_and_tags_and_labels(filename);
    std::string centers_file(filename);
    std::string index_file(filename);
    centers_file += "_centers.bin";
    _u64 tmp_dim1, tmp_dim2;  // should be same as aligned_dim
    grann::load_aligned_bin<float>(centers_file, _cluster_centers,
                                   _num_clusters, tmp_dim1, tmp_dim2);

    std::ifstream in(index_file, std::ios::binary | std::ios::in);
    _u64          total_count = 0;

    _inverted_index.resize(_num_clusters);
    for (unsigned i = 0; i < this->_num_clusters; i++) {
      unsigned GK;
      in.read((char *) &GK, sizeof(unsigned));
      _inverted_index[i].resize(GK);
      in.read((char *) _inverted_index[i].data(), GK * sizeof(unsigned));
      total_count += GK;
    }
    in.close();
    std::cout << "Read a total of " << total_count
                << " points from inverted index file." << std::endl;
  }

  template<typename T>
  void IVFIndex<T>::build(const Parameters &build_params) {
    _num_clusters = build_params.Get<_u32>("num_clusters");
    float training_prob = build_params.Get<float>("training_rate");

    _inverted_index.resize(_num_clusters);
    _cluster_centers = new float[_num_clusters * this->_aligned_dim];

    float *train_data_float;
    _u64   num_train;

    gen_random_slice<T>(this->_data, this->_num_points, this->_aligned_dim,
                        training_prob, train_data_float, num_train);
    std::cout << "Going to train the cluster centers over " << num_train
                << " points " << std::endl;

    math_utils::kmeans_plus_plus_centers(train_data_float, num_train,
                                         this->_aligned_dim, _cluster_centers,
                                         _num_clusters);

    math_utils::run_lloyds(train_data_float, num_train, this->_aligned_dim,
                           _cluster_centers, _num_clusters, MAX_K_MEANS_REPS,
                           nullptr, nullptr);
    std::cout << "Done. Now going to build the index." << std::endl;

    std::vector<_u32> closest_centers(this->_num_points);
    float *data_float;
    bool local_data = false;

    if (sizeof(T) == sizeof(float)) {
      data_float = (float*) (this->_data);
    } else
    {
    data_float = new float[this->_num_points * this->_aligned_dim];  
    grann::convert_types(this->_data, data_float, this->_num_points,
                         this->_aligned_dim);
    local_data = true;
    }
    math_utils::compute_closest_centers(
        data_float, this->_num_points, this->_aligned_dim, _cluster_centers,
        _num_clusters, 1, closest_centers.data(), _inverted_index.data(),
        nullptr);

    delete[] train_data_float;
    if (local_data)
      delete[] data_float;
  }

  // returns # results found (will be <= res_count)
  template<typename T>
  _u32 IVFIndex<T>::search(const T *query, _u32 res_count,
                           const Parameters &search_params, _u32 *indices,
                           float *distances, QueryStats *stats,
													 std::vector<label> search_filters) {
    _u32 res_cnt = 0;
    _u32 probe_width = search_params.Get<_u32>("probe_width");

    float *query_float = new float[this->_aligned_dim];
    grann::convert_types(query, query_float, 1, this->_aligned_dim);

    std::vector<_u32> closest_centers(probe_width, 0);
    math_utils::compute_closest_centers(
        query_float, 1, this->_aligned_dim, _cluster_centers, _num_clusters,
        probe_width, closest_centers.data(), nullptr, nullptr);

    std::vector<_u32> candidates;
    for (auto &x : closest_centers) {
      candidates.insert(candidates.end(), _inverted_index[x].begin(),
                        _inverted_index[x].end());
    }
    std::vector<Neighbor> best_candidates(res_count + 1);
    _u32                  cur_size = 0;
    _u32                  max_size = res_count;
    _u32                  cmps = 0;
    tsl::robin_set<_u32>  inserted;
    ANNIndex<T>::process_candidates_into_best_candidates_pool(
        query, candidates, best_candidates, max_size, 
				cur_size, inserted, cmps, search_filters);

    res_cnt = cur_size < res_count ? cur_size : res_count;

    for (_u32 i = 0; i < res_cnt; i++) {
      indices[i] = best_candidates[i].id;
      if (distances != nullptr) {
        distances[i] = best_candidates[i].distance;
      }
    }
    if (stats != nullptr) {
      stats->n_cmps += cmps;
    }

    delete[] query_float;
    return res_cnt;
  }

  // EXPORTS
  template class IVFIndex<float>;
  template class IVFIndex<int8_t>;
  template class IVFIndex<uint8_t>;
}  // namespace grann
