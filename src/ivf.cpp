// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "ivf.h"
#include "partition_and_pq.h"
#include "math_utils.h"
#include "boost/algorithm/string.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <set>

namespace grann {

  // Initialize a generic graph-based index with metric m, load the data of type
  // T with filename (bin)
  template<typename T>
  IVFIndex<T>::IVFIndex(Metric m, const char *filename,
                        std::vector<_u32> &list_of_tags,
                        std::string        labels_fname)
      : ANNIndex<T>(m, filename, list_of_tags,
                    labels_fname) {  // Graph Index class constructor loads the
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

		std::ifstream freq_in("something_frequency_table.txt");
		std::string line;
		std::getline(freq_in, line); //first line is center name
		for (_u32 x = 0; x < _num_clusters; x++) {
			while(getline(freq_in, line)) {
				if (line[0] == 'C') {
					break;
				}
				std::vector<label> results;
				boost::algorithm::split(results, line, boost::is_any_of(":"));
				_frequency_table[x][results[0]] = std::stoi(results[1]); 
			}
		}
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
    float *data_float = new float[this->_num_points * this->_aligned_dim];

    grann::convert_types(this->_data, data_float, this->_num_points,
                         this->_aligned_dim);
    math_utils::compute_closest_centers(
        data_float, this->_num_points, this->_aligned_dim, _cluster_centers,
        _num_clusters, 1, closest_centers.data(), _inverted_index.data(),
        nullptr);
    // std::set<_u32> closest_c(closest_centers.begin(), closest_centers.end());
    for (_u32 x = 0; x < _num_clusters; x++) {
      /*std::cout << "size of inverted index of " << x << " is "
                << _inverted_index[x].size() << std::endl;*/
      for (auto &y : _inverted_index[x]) {
        // std::cout << "size of pts_to_labels of " << y << " is "
        //<< this->_pts_to_labels[y].size() << std::endl;
        for (auto &z : this->_pts_to_labels[y]) {
          /*if (z == "21")
            std::cout << "before for " << x << "," << y << "," << z << " "
                      << _frequency_table[x][z] << std::endl;*/
          _frequency_table[x][z]++;
          /*if (z == "21")
            std::cout << "after for " << x << "," << y << "," << z << " "
                      << _frequency_table[x][z] << std::endl;*/
        }
      }
    }
    std::ofstream myfile;
    myfile.open("something_frequency_table.txt");
    // myfile << "Writing the frequencies of labels in clusters.\n";
    for (_u32 x = 0; x < _num_clusters; x++) {
      myfile << "Center " << x << std::endl;
      for (auto &z : _frequency_table[x]) {
        myfile << z.first << ":" << z.second << std::endl;
      }
    }
    delete[] train_data_float;
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
		_u32 pw_hyperparam = 3;

    float *query_float = new float[this->_aligned_dim];
    grann::convert_types(query, query_float, 1, this->_aligned_dim);

    std::vector<_u32> closest_centers(probe_width, 0);
    math_utils::compute_closest_centers(
        query_float, 1, this->_aligned_dim, _cluster_centers, _num_clusters,
      	probe_width, closest_centers.data(), nullptr, nullptr);

    std::vector<_u32> candidates;
    for (auto &x : closest_centers) {
			_u8 skip = 0;
			/*
			for (auto const &y : search_filters) {
				if (_frequency_table[x][y] == 0) {
					skip = 1;
					break;
				}
			}
			*/
			if (skip) continue;
      candidates.insert(candidates.end(), _inverted_index[x].begin(),
                        _inverted_index[x].end());
    }
    std::vector<Neighbor> best_candidates(res_count + 1);
    _u32                  cur_size = 0;
    _u32                  max_size = res_count;
    _u32                  cmps = 0;
    tsl::robin_set<_u32>  inserted;
    ANNIndex<T>::process_candidates_into_best_candidates_pool(
        query, candidates, best_candidates, max_size, cur_size, inserted, cmps,
        search_filters);

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
