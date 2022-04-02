// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "recall_utils.h"
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


    std::ifstream infile(filename);
    std::string   line, token;
    unsigned      line_cnt = 0;
    _u32 total_pts = 0;
    while (std::getline(infile, line)) {
      std::istringstream       iss(line);
      std::vector<_u32> lbls(0);

      while (getline(iss, token, ',')) {
        token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
        lbls.push_back(std::atoi(token.c_str()));
        total_pts++;
      }
      if (lbls.size() <= 0) {
        std::cout << "No label found";
        exit(-1);
      }
//      std::sort(lbls.begin(), lbls.end()); // labels are sorted to do correct set_intersection (if needed)
      _inverted_index.push_back(lbls);
      line_cnt++;
    }

    std::cout<<"Read " << line_cnt << " clusters worth data from inverted index file." << std::endl;
    std::cout<<"Read " << total_pts << " points (with possible multiplicity) across all clusters in inverted index file." << std::endl;

    std::string truthfile (filename);
    truthfile += "_gt.bin";
    grann::load_truthset(truthfile, _gtids, _gtdists, _gtnum, _gtdim);

    _u64 nr, nd;
    std::string reorder_file;
    reorder_file = std::string(filename) +  "_id_to_loc.bin";
    if (file_exists(reorder_file)) {
      grann::load_bin<_u32> (reorder_file, _id_to_location, nr, nd);
    } else {
      _id_to_location = new _u32[this->_num_clusters];
      std::iota(_id_to_location, _id_to_location + this->_num_clusters, 0);
    }

    reorder_file = std::string(filename) +  "_loc_to_id.bin";
    if (file_exists(reorder_file)) {
      grann::load_bin<_u32> (reorder_file, _location_to_id, nr, nd);
    } else {
            _location_to_id = new _u32[this->_num_clusters];
      std::iota(_location_to_id, _location_to_id + this->_num_clusters, 0);
    }

    for (_u32 i = 0; i < this->_num_clusters; i++) {
      if (_location_to_id[_id_to_location[i]] != i) {
        std::cout<<"Error! Exitting." << std::endl;
        exit(-1);
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
    _u32 idx = search_params.Get<_u32>("idx");
    int32_t num_nearby = search_params.Get<_u32>("num_nearby");

    probe_width = probe_width > _gtdim ? _gtdim : probe_width;

    float *query_float = new float[this->_aligned_dim];
    grann::convert_types(query, query_float, 1, this->_aligned_dim);

    std::vector<_u32> closest_centers(_gtdim, 0);

    std::memcpy (closest_centers.data(), _gtids + idx*_gtdim, _gtdim * sizeof(_u32));
/*    math_utils::compute_closest_centers(
        query_float, 1, this->_aligned_dim, _cluster_centers, _num_clusters,
        probe_width, closest_centers.data(), nullptr, nullptr);
        */
    
    _u32 io_cnt = 0;
    tsl::robin_set<_u32> seen_pages;
    std::vector<_u32> candidates;
    for (auto &x : closest_centers) {
      bool fetch_flag = false;
//      candidates.insert(candidates.end(), _inverted_index[x].begin(),
//                        _inverted_index[x].end());
      if (_id_to_location != nullptr) {
        auto y = _id_to_location[x];
        for (int32_t nearby_ids = (int32_t)y - num_nearby; nearby_ids <= (int32_t)y + num_nearby; nearby_ids++) {
            if (nearby_ids < 0)
            continue;
            if (nearby_ids >= this->_num_clusters)
            break;
            auto z = _location_to_id[nearby_ids];
            if (seen_pages.find(z) == seen_pages.end()) {
            if (fetch_flag == false) {
              fetch_flag = true;
              io_cnt++;
            }
            candidates.insert(candidates.end(), _inverted_index[z].begin(),
                        _inverted_index[z].end());
            seen_pages.insert(z);
            }
        }
      }
            if (io_cnt >= probe_width)
            break;
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
