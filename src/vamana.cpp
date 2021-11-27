// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>

#include "utils.h"
#include "vamana.h"

namespace grann {

  // Initialize an vamana with metric m, load the data of type T with filename
  // (bin), and initialize num_points
  template<typename T>
  Vamana<T>::Vamana(Metric m, const char *filename,
                    std::vector<_u32> &list_of_tags)
      : GraphIndex<T>(m, filename, list_of_tags) {
    grann::cout << "Initialized Vamana Object with " << this->_num_points
                << " points, dim=" << this->_dim << "." << std::endl;
  }

  template<typename T>
  Vamana<T>::Vamana(Metric m)
      : GraphIndex<T>(m) {
    grann::cout << "Initialized Empty Vamana Object."<< std::endl;
  }


  // save the graph vamana on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T>
  void Vamana<T>::save(const char *filename) {

    ANNIndex<T>::save_data_and_tags(filename);
    long long     total_gr_edges = 0;
    _u64          vamana_size = 0;
    std::ofstream out(std::string(filename), std::ios::binary | std::ios::out);

    out.write((char *) &vamana_size, sizeof(uint64_t));
    out.write((char *) &this->_max_degree, sizeof(unsigned));
    out.write((char *) &this->_start_node, sizeof(unsigned));
    for (unsigned i = 0; i < this->_num_points; i++) {
      unsigned GK = (unsigned) this->_out_nbrs[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) this->_out_nbrs[i].data(), GK * sizeof(unsigned));
      total_gr_edges += GK;
    }
    vamana_size = out.tellp();
    out.seekp(0, std::ios::beg);
    out.write((char *) &vamana_size, sizeof(uint64_t));
    out.close();
  }

  // load the vamana from file and update the width (max_degree), ep
  // (navigating node id), and _out_nbrs (adjacency list)
  template<typename T>
  void Vamana<T>::load(const char *filename) {
    ANNIndex<T>::load_data_and_tags(filename);
    std::ifstream in(filename, std::ios::binary);
    _u64          expected_file_size;
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &this->_max_degree, sizeof(unsigned));
    in.read((char *) &this->_start_node, sizeof(unsigned));
    grann::cout << "Loading vamana index " << filename << "..." << std::flush;

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

    grann::cout << "..done. Vamana has " << nodes << " nodes and " << cc
                << " out-edges" << std::endl;
  }

  /**************************************************************
   *      Support for Static Vamana Building and Searching
   **************************************************************/


  template<typename T>
  void Vamana<T>::get_expanded_nodes(
      const _u64 node_id, const unsigned l_build,
      std::vector<unsigned>     init_ids,
      std::vector<Neighbor> &   expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids) {
    const T *node_coords = this->_data + this->_aligned_dim * node_id;
    std::vector<Neighbor> best_L_nodes;

    if (init_ids.size() == 0)
      init_ids.emplace_back(this->_start_node);

    this->greedy_search_to_fixed_point(node_coords, l_build, init_ids,
                                       expanded_nodes_info, expanded_nodes_ids,
                                       best_L_nodes);
  }


  template<typename T>
  void Vamana<T>::build(Parameters &build_parameters) {
    grann::Timer build_timer;

    unsigned num_threads = build_parameters.Get<unsigned>("num_threads");
    unsigned L = build_parameters.Get<unsigned>("L");
    unsigned degree_bound = build_parameters.Get<unsigned>("R");
    float    alpha = build_parameters.Get<float>("alpha");

    grann::cout << "Starting vamana build with listSize L=" << L
                << ", degree bound R=" << degree_bound
                << ", and alpha=" << alpha << std::endl;

    this->_locks_enabled =
        true;  // we dont need locks for pure search on a pre-built index
    this->_locks = std::vector<std::mutex>(this->_num_points);
    this->_out_nbrs.resize(this->_num_points);
    for (auto &x : this->_out_nbrs)
      x.reserve(1.05 * VAMANA_SLACK_FACTOR * degree_bound);

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    this->_start_node = ANNIndex<T>::calculate_entry_point();
    grann::cout << "Medoid identified as " << this->_start_node << std::endl;

    _u32             progress_milestone = (_u32)(this->_num_points / 10);
    std::atomic<int> milestone_marker{0};

#pragma omp parallel for schedule(static, 64)
    for (_u32 location = 0; location < this->_num_points; location++) {
      if (location % progress_milestone == 0) {
        ++milestone_marker;

        std::stringstream msg;
        msg << (milestone_marker * 10) << "\% of build completed \r";
        grann::cout << msg.str();
      }

      std::vector<Neighbor> pool;
      std::vector<Neighbor> tmp;
      tsl::robin_set<_u32>  visited;
      pool.reserve(2 * L);
      tmp.reserve(2 * L);
      visited.reserve(20 * L);

      std::vector<_u32> pruned_list;
      std::vector<_u32> init_ids;
      get_expanded_nodes(location, L, init_ids, pool, visited);

      this->prune_candidates_alpha_rng(location, pool, build_parameters, pruned_list);

      this->_out_nbrs[location].reserve(
          (_u64)(VAMANA_SLACK_FACTOR * degree_bound));
      {
        LockGuard guard(this->_locks[location]);
        for (auto link : pruned_list)
          this->_out_nbrs[location].emplace_back(link);
      }
      GraphIndex<T>::add_reciprocal_edges(location, pruned_list,
                   build_parameters);  // add reverse edges
    }
    grann::cout << "Starting final cleanup.." << std::flush;
#pragma omp parallel for schedule(dynamic, 65536)
    for (_u64 node = 0; node < this->_num_points; node++) {
      if (this->_out_nbrs[node].size() > degree_bound) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);
        std::vector<unsigned>    new_out_neighbors;

        for (auto cur_nbr : this->_out_nbrs[node]) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != node) {
            float dist = this->_distance->compare(
                this->_data + this->_aligned_dim * (_u64) node,
                this->_data + this->_aligned_dim * (_u64) cur_nbr,
                (unsigned) this->_aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        this->prune_candidates_alpha_rng(node, dummy_pool, build_parameters,
                              new_out_neighbors);

        this->_out_nbrs[node].clear();
        for (auto id : new_out_neighbors)
          this->_out_nbrs[node].emplace_back(id);
      }
    }

    grann::cout << "done." << std::endl;
    this->_has_built = true;
    this->update_degree_stats();

    grann::cout << "Total build time: "
                << ((double) build_timer.elapsed() / (double) 1000000) << "s"
                << std::endl;
  }

  template<typename T>
  _u32 Vamana<T>::search(const T *query, _u32 res_count,
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
  template class Vamana<float>;
  template class Vamana<int8_t>;
  template class Vamana<uint8_t>;
}  // namespace grann
