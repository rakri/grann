// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>

#include "utils.h"
#include "relative_ng.h"

namespace grann {

  // Initialize an rng with metric m, load the data of type T with filename
  // (bin), and initialize num_points
  template<typename T>
  RelativeNG<T>::RelativeNG(Metric m, const char *filename,
                    std::vector<_u32> &list_of_tags)
      : GraphIndex<T>(m, filename, list_of_tags) {
    grann::cout << "Initialized RelativeNG Object with " << this->_num_points
                << " points, dim=" << this->_dim << "." << std::endl;
  }

  // save the graph rng on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T>
  void RelativeNG<T>::save(const char *filename) {
    ANNIndex<T>::save_data_and_tags(filename);
    long long     total_gr_edges = 0;
    _u64          rng_size = 0;
    std::ofstream out(std::string(filename), std::ios::binary | std::ios::out);

    out.write((char *) &rng_size, sizeof(uint64_t));
    out.write((char *) &this->_max_degree, sizeof(unsigned));
    out.write((char *) &this->_start_node, sizeof(unsigned));
    for (unsigned i = 0; i < this->_num_points; i++) {
      unsigned GK = (unsigned) this->_out_nbrs[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) this->_out_nbrs[i].data(), GK * sizeof(unsigned));
      total_gr_edges += GK;
    }
    rng_size = out.tellp();
    out.seekp(0, std::ios::beg);
    out.write((char *) &rng_size, sizeof(uint64_t));
    out.close();
  }

  // load the rng from file and update the width (max_degree), ep
  // (navigating node id), and _out_nbrs (adjacency list)
  template<typename T>
  void RelativeNG<T>::load(const char *filename) {
    ANNIndex<T>::load_data_and_tags(filename);
    std::ifstream in(filename, std::ios::binary);
    _u64          expected_file_size;
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &this->_max_degree, sizeof(unsigned));
    in.read((char *) &this->_start_node, sizeof(unsigned));
    grann::cout << "Loading rng index " << filename << "..." << std::flush;

    _u64     cc = 0;
    unsigned nodes = 0;
    while (!in.eof()) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (in.eof())
        break;
      cc += k;
      ++nodes;
      std::vector<unsigned> tmp(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));

      this->_out_nbrs.emplace_back(tmp);
      if (nodes % 10000000 == 0)
        grann::cout << "." << std::flush;
    }
    if (this->_out_nbrs.size() != this->_num_points) {
      grann::cout << "ERROR. mismatch in number of points. Graph has "
                  << this->_out_nbrs.size() << " points and loaded dataset has "
                  << this->_num_points << " points. " << std::endl;
      return;
    }

    grann::cout << "..done. RelativeNG has " << nodes << " nodes and " << cc
                << " out-edges" << std::endl;
  }

  /**************************************************************
   *      Support for Static RelativeNG Building and Searching
   **************************************************************/

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T>
  unsigned RelativeNG<T>::calculate_entry_point() {
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
      float &  dist = distances[i];
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

  template<typename T>
  void RelativeNG<T>::build(Parameters &build_parameters) {
    grann::Timer build_timer;

    unsigned num_threads = build_parameters.Get<unsigned>("num_threads");

    grann::cout << "Starting rng build." << std::endl;

//    this->_locks_enabled =
//        true;  // we dont need locks for pure search on a pre-built index
//    this->_locks = std::vector<std::mutex>(this->_num_points);

    this->_out_nbrs.resize(this->_num_points);

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    this->_start_node = calculate_entry_point();
    grann::cout << "Medoid identified as " << this->_start_node << std::endl;

    _u32             progress_milestone = (_u32)(this->_num_points / 10);
    std::atomic<int> milestone_marker{0};

      build_parameters.Set<_u32>("C", this->_num_points);
      build_parameters.Set<_u32>("R", this->_num_points);
      build_parameters.Set<float>("alpha", 1);
      build_parameters.Set<_u32>("L", this->_num_points);


#pragma omp parallel for schedule(static, 64)
    for (_u32 location = 0; location < this->_num_points; location++) {
      if (location % progress_milestone == 0) {
        ++milestone_marker;

        std::stringstream msg;
        msg << (milestone_marker * 10) << "\% of build completed" << std::endl;
        grann::cout << msg.str();
      }

      std::vector<Neighbor> pool;
      std::vector<_u32> pruned_list;


      for (_u32 j = 0; j < this->_num_points; j++) {
        if (j == location)
        continue;
        float dist = this->_distance->compare(this->_data + this->_aligned_dim* (_u64) j, this->_data + this->_aligned_dim* (_u64) location, this->_aligned_dim);
        pool.emplace_back(Neighbor(j, dist, true));
      }

      this->prune_neighbors(location, pool, build_parameters, pruned_list);

      this->_out_nbrs[location].reserve(pruned_list.size());
        for (auto link : pruned_list)
          this->_out_nbrs[location].emplace_back(link);

    }

    grann::cout << "done." << std::endl;
    this->_has_built = true;
    this->update_degree_stats();

    grann::cout << "Total build time: "
                << ((double) build_timer.elapsed() / (double) 1000000) << "s"
                << std::endl;
  }

  template<typename T>
  _u32 RelativeNG<T>::search(const T *query, _u32 res_count,
                         Parameters &search_params, _u32 *indices,
                         float *distances, QueryStats *stats) {
    _u32                     search_list_size = search_params.Get<_u32>("L");
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * search_list_size);
    std::vector<Neighbor>    top_candidate_list, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    init_ids.emplace_back(this->_start_node);

    auto algo_fetched_count = this->greedy_search_to_fixed_point(
        query, search_list_size, init_ids, expanded_nodes_info,
        expanded_nodes_ids, top_candidate_list, stats);

    //     _u64 pos = 0;
    for (_u32 i = 0; i < res_count; i++) {
      if (i >= res_count)
        break;
      indices[i] = this->_tag_map[top_candidate_list[i].id];
      distances[i] = top_candidate_list[i].distance;
    }
    return std::min(res_count, algo_fetched_count);
  }

  // EXPORTS
  template class RelativeNG<float>;
  template class RelativeNG<int8_t>;
  template class RelativeNG<uint8_t>;
}  // namespace grann
