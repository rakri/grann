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
                    std::vector<_u32> &list_of_ids)
      : GraphIndex<T>(m, filename, list_of_ids) {
    grann::cout << "Initialized Vamana Object with " << this->_num_points
                << " points, dim=" << this->_dim << std::endl;
  }

  // save the graph vamana on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T>
  void Vamana<T>::save(const char *filename) {
    long long     total_gr_edges = 0;
    size_t        vamana_size = 0;
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

    grann::cout << "Avg degree: "
                << ((float) total_gr_edges) / ((float) (this->_num_points))
                << std::endl;
  }

  // load the vamana from file and update the width (max_degree), ep
  // (navigating node id), and _out_nbrs (adjacency list)
  template<typename T>
  void Vamana<T>::load(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    size_t        expected_file_size;
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &this->_max_degree, sizeof(unsigned));
    in.read((char *) &this->_start_node, sizeof(unsigned));
    grann::cout << "Loading vamana index " << filename << "..." << std::flush;

    size_t   cc = 0;
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

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T>
  unsigned Vamana<T>::calculate_entry_point() {
    // allocate and init centroid
    float *center = new float[this->_aligned_dim]();
    for (size_t j = 0; j < this->_aligned_dim; j++)
      center[j] = 0;

    for (size_t i = 0; i < this->_num_points; i++)
      for (size_t j = 0; j < this->_aligned_dim; j++)
        center[j] += this->_data[i * this->_aligned_dim + j];

    for (size_t j = 0; j < this->_aligned_dim; j++)
      center[j] /= this->_num_points;

    // compute all to one distance
    float *distances = new float[this->_num_points]();
#pragma omp parallel for schedule(static, 65536)
    for (_s64 i = 0; i < (_s64) this->_num_points; i++) {
      // extract point and distance reference
      float &  dist = distances[i];
      const T *cur_vec = this->_data + (i * (size_t) this->_aligned_dim);
      dist = 0;
      float diff = 0;
      for (size_t j = 0; j < this->_aligned_dim; j++) {
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
  void Vamana<T>::get_expanded_nodes(
      const size_t node_id, const unsigned l_build,
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
  void Vamana<T>::occlude_list(std::vector<Neighbor> &pool, const float alpha,
                               const unsigned degree, const unsigned maxc,
                               std::vector<Neighbor> &result) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    auto               pool_size = (_u32) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);

    unsigned start = 0;
    while (result.size() < degree && (start) < pool.size() && start < maxc) {
      auto &p = pool[start];
      if (occlude_factor[start] > alpha) {
        start++;
        continue;
      }
      occlude_factor[start] = std::numeric_limits<float>::max();
      result.push_back(p);
      for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
        if (occlude_factor[t] > alpha)
          continue;
        float djk = this->_distance->compare(
            this->_data + this->_aligned_dim * (size_t) pool[t].id,
            this->_data + this->_aligned_dim * (size_t) p.id,
            (unsigned) this->_aligned_dim);
        occlude_factor[t] =
            (std::max)(occlude_factor[t], pool[t].distance / djk);
      }
      start++;
    }
  }

  template<typename T>
  void Vamana<T>::prune_neighbors(const unsigned         location,
                                  std::vector<Neighbor> &pool,
                                  const Parameters &     parameter,
                                  std::vector<unsigned> &pruned_list) {
    unsigned degree_bound = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float    alpha = parameter.Get<float>("alpha");

    if (pool.size() == 0)
      return;

    //_max_degree = (std::max)(this->_max_degree, degree_bound);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(degree_bound);

    occlude_list(pool, alpha, degree_bound, maxc, result);

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    pruned_list.clear();
    assert(result.size() <= degree_bound);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
    }
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T>
  void Vamana<T>::inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                               const Parameters &parameters) {
    const auto degree_bound = parameters.Get<unsigned>("R");

    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      /* des_pool contains the neighbors of the neighbors of n */
      auto &                des_pool = this->_out_nbrs[des];
      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        LockGuard guard(this->_locks[des]);
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < VAMANA_SLACK_FACTOR * degree_bound) {
            des_pool.emplace_back(n);
            prune_needed = false;
          } else {
            copy_of_neighbors = des_pool;
            prune_needed = true;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        copy_of_neighbors.push_back(n);
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);

        size_t reserveSize =
            (size_t)(std::ceil(1.05 * VAMANA_SLACK_FACTOR * degree_bound));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != des) {
            float dist = this->_distance->compare(
                this->_data + this->_aligned_dim * (size_t) des,
                this->_data + this->_aligned_dim * (size_t) cur_nbr,
                (unsigned) this->_aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        prune_neighbors(des, dummy_pool, parameters, new_out_neighbors);
        {
          LockGuard guard(this->_locks[des]);
          this->_out_nbrs[des].clear();
          for (auto new_nbr : new_out_neighbors) {
            this->_out_nbrs[des].emplace_back(new_nbr);
          }
        }
      }
    }
  }

  template<typename T>
  void Vamana<T>::build(Parameters &build_parameters) {
    grann::cout << "Starting vamana build over " << this->_num_points
                << " points in " << this->_dim << " dims." << std::endl;
    grann::Timer build_timer;

    unsigned num_threads = build_parameters.Get<unsigned>("num_threads");
    unsigned L = build_parameters.Get<unsigned>("L");
    unsigned degree_bound = build_parameters.Get<unsigned>("R");

    this->_locks_enabled =
        true;  // we dont need locks for pure search on a pre-built index
    this->_locks = std::vector<std::mutex>(this->_num_points);
    this->_out_nbrs.resize(this->_num_points);
    for (auto &x : this->_out_nbrs)
      x.reserve(1.05 * VAMANA_SLACK_FACTOR * degree_bound);

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    this->_start_node = calculate_entry_point();

    _u32             progress_milestone = (_u32)(this->_num_points / 20);
    std::atomic<int> milestone_marker{0};

#pragma omp parallel for schedule(static, 64)
    for (_u32 location = 0; location < this->_num_points; location++) {
      if (location % progress_milestone == 0) {
        ++milestone_marker;

        std::stringstream msg;
        msg << (milestone_marker * 5) << "\% of build completed" << std::endl;
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

      prune_neighbors(location, pool, build_parameters, pruned_list);

      this->_out_nbrs[location].reserve(
          (_u64)(VAMANA_SLACK_FACTOR * degree_bound));
      {
        LockGuard guard(this->_locks[location]);
        for (auto link : pruned_list)
          this->_out_nbrs[location].emplace_back(link);
      }
      inter_insert(location, pruned_list,
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
                this->_data + this->_aligned_dim * (size_t) node,
                this->_data + this->_aligned_dim * (size_t) cur_nbr,
                (unsigned) this->_aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        prune_neighbors(node, dummy_pool, build_parameters, new_out_neighbors);

        this->_out_nbrs[node].clear();
        for (auto id : new_out_neighbors)
          this->_out_nbrs[node].emplace_back(id);
      }
    }

    grann::cout << "done.." << std::endl;
    this->_has_built = true;
    this->update_degree_stats();

    grann::cout << "Build completed in time: "
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

    //     size_t pos = 0;
    for (_u32 i = 0; i < res_count; i++) {
      if (i >= res_count)
        break;
      indices[i] = this->idmap[top_candidate_list[i].id];
      distances[i] = top_candidate_list[i].distance;
    }
    return std::min(res_count, algo_fetched_count);
  }

  // EXPORTS
  template GRANN_DLLEXPORT class Vamana<float>;
  template GRANN_DLLEXPORT class Vamana<int8_t>;
  template GRANN_DLLEXPORT class Vamana<uint8_t>;
}  // namespace grann
