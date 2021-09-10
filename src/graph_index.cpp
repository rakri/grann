// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "graph_index.h"
#include <iomanip>

namespace grann {

  template<typename T>
  _u32 GraphIndex<T>::process_neighbors_into_candidate_pool(
      const T *&node_coords, std::vector<_u32> &nbr_list,
      std::vector<Neighbor> &best_L_nodes, const _u32 maxListSize,
      _u32 &curListSize, tsl::robin_set<_u32> &inserted_into_pool,
      _u32 &total_comparisons) {
    _u32 best_inserted_position = maxListSize;
    for (unsigned m = 0; m < nbr_list.size(); ++m) {
      unsigned id = nbr_list[m];
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);

        if ((m + 1) < nbr_list.size()) {
          auto nextn = nbr_list[m + 1];
          grann::prefetch_vector(
              (const char *) this->_data + this->_aligned_dim * (size_t) nextn,
              sizeof(T) * this->_aligned_dim);
        }

        total_comparisons++;
        float dist = this->_distance->compare(
            node_coords, this->_data + this->_aligned_dim * (size_t) id,
            (unsigned) this->_aligned_dim);

        if (dist >= best_L_nodes[curListSize - 1].distance &&
            (curListSize == maxListSize))
          continue;

        Neighbor nn(id, dist, true);
        unsigned r = InsertIntoPool(best_L_nodes.data(), curListSize, nn);
        if (curListSize < maxListSize)
          ++curListSize;  // pool has grown by +1
        if (r < best_inserted_position)
          best_inserted_position = r;
      }
    }
    return best_inserted_position;
  }

  // Initialize a generic graph-based index with metric m, load the data of type
  // T with filename (bin)
  template<typename T>
  GraphIndex<T>::GraphIndex(Metric m, const char *filename,
                            std::vector<_u32> &list_of_ids)
      : ANNIndex<T>(m, filename,
                    list_of_ids) {  // Graph Index class constructor loads the
                                    // data and sets num_points, dim, etc.
    _max_degree = 0;
  }

  /* greedy_search_to_fixed_point():
   * node_coords : point whose neighbors to be found.
   * init_ids : ids of initial search list.
   * Lsize : size of list.
   * beam_max_degree: beam_max_degree when performing vamanaing
   * expanded_nodes_info: will contain all the node ids and distances from
   * query that are expanded
   * expanded_nodes_ids : will contain all the nodes that are expanded during
   * search.
   * best_L_nodes: ids of closest L nodes in list
   */
  template<typename T>
  _u32 GraphIndex<T>::greedy_search_to_fixed_point(
      const T *node_coords, const unsigned Lsize,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor> &      expanded_nodes_info,
      tsl::robin_set<unsigned> &   expanded_nodes_ids,
      std::vector<Neighbor> &best_L_nodes, QueryStats *stats) {
    best_L_nodes.resize(Lsize + 1);
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_ids.reserve(10 * Lsize);

    unsigned                 l = 0;
    Neighbor                 nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    for (auto id : init_ids) {
      nn = Neighbor(id,
                    this->_distance->compare(
                        this->_data + this->_aligned_dim * (size_t) id,
                        node_coords, (unsigned) this->_aligned_dim),
                    true);
      if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
        inserted_into_pool.insert(id);
        best_L_nodes[l++] = nn;
      }
      if (l == Lsize)
        break;
    }

    /* sort best_L_nodes based on distance of each point to node_coords */
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;
    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (k < l) {
      // unsigned nk = l;

      if (best_L_nodes[k].flag) {
        hops++;
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        expanded_nodes_info.emplace_back(best_L_nodes[k]);
        expanded_nodes_ids.insert(n);

        std::vector<unsigned> des;
        if (_locks_enabled) {
          LockGuard guard(_locks[n]);
          des = _out_nbrs[n];
        }
        unsigned best_inserted_index;
        if (_locks_enabled)
          best_inserted_index = process_neighbors_into_candidate_pool(
              node_coords, des, best_L_nodes, Lsize, l, inserted_into_pool,
              cmps);
        else
          best_inserted_index = process_neighbors_into_candidate_pool(
              node_coords, _out_nbrs[n], best_L_nodes, Lsize, l,
              inserted_into_pool, cmps);

        if (best_inserted_index <= k)
          k = best_inserted_index;
        else
          ++k;
      } else
        k++;
    }
    if (stats != nullptr) {
      stats->n_hops = hops;
      stats->n_cmps = cmps;
    }
    return l;
  }

  template<typename T>
  void GraphIndex<T>::update_degree_stats() {
    if (!this->_has_built)
      return;

    this->_max_degree = 0;
    std::vector<_u32>                  out_degrees(this->_num_points, 0);
    std::vector<std::pair<_u32, _u32>> in_degrees(this->_num_points);
    for (_u32 i = 0; i < this->_num_points; i++) {
      in_degrees[i].first = i;
      in_degrees[i].second = 0;
    }

    for (_u32 i = 0; i < this->_num_points; i++) {
      this->_max_degree = this->_max_degree > this->_out_nbrs[i].size()
                              ? this->_max_degree
                              : this->_out_nbrs[i].size();
      out_degrees[i] = this->_out_nbrs[i].size();
      for (auto &x : this->_out_nbrs[i]) {
        in_degrees[x].second++;
      }
    }

    std::sort(in_degrees.begin(), in_degrees.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.second < rhs.second;
              });

    // std::sort(in_degrees.begin(), in_degrees.end());
    std::sort(out_degrees.begin(), out_degrees.end());

    _u32 unreachable_count = 0;
    while (unreachable_count < this->_num_points &&
           (in_degrees[unreachable_count].second == 0)) {
      unreachable_count++;
    }

    grann::cout << std::setw(16) << "Percentile" << std::setw(16) << "Out Degree"
              << std::setw(16) << "In Degree" << std::endl;
    grann::cout << "======================================================="
              << std::endl;
    for (_u32 p = 0; p < 100; p += 10) {
      grann::cout << std::setw(16) << p << std::setw(16)
                << out_degrees[(_u64)((p / 100.0) * this->_num_points)]
                << std::setw(16)
                << in_degrees[(_u64)((p / 100.0) * this->_num_points)].second
                << std::endl;
    }
    grann::cout << std::setw(16) << "100" << std::setw(16)
              << out_degrees[this->_num_points - 1] << std::setw(16)
              << in_degrees[this->_num_points - 1].second << std::endl;

    grann::cout << std::setprecision(3)
              << (100.0 * unreachable_count) / this->_num_points
              << "\% points are unreachable." << std::endl;
    grann::cout << in_degrees[this->_num_points - 1].first
              << " is the most popular in-degree node." << std::endl;
  }

  // EXPORTS
  template  class GraphIndex<float>;
  template  class GraphIndex<int8_t>;
  template  class GraphIndex<uint8_t>;
}  // namespace grann