// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "graph_index.h"
#include <iomanip>
#include <sstream>

namespace grann {

  // Initialize a generic graph-based index with metric m, load the data of type
  // T with filename (bin)
  template<typename T>
  GraphIndex<T>::GraphIndex(Metric m, const char *filename,
                            std::vector<_u32> &list_of_tags,
                        std::string        labels_fname)
      : ANNIndex<T>(m, filename,
                    list_of_tags, labels_fname) {  // Graph Index class constructor loads the
                                     // data and sets num_points, dim, etc.
    _max_degree = 0;
  }

  template<typename T>
  GraphIndex<T>::GraphIndex(Metric m)
      : ANNIndex<T>(m) {  // Graph Index class constructor empty for load.
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
      std::vector<Neighbor> &best_L_nodes, const std::vector<label> &labels_to_filter_by, QueryStats *stats) {
    best_L_nodes.resize(Lsize + 1);
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_ids.reserve(10 * Lsize);


    unsigned                 l = 0;
    Neighbor                 nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    for (auto id : init_ids) {
      nn = Neighbor(
          id,
          this->_distance->compare(this->_data + this->_aligned_dim * (_u64) id,
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
          ReadLock guard(_locks[n]);
          des = _out_nbrs[n];
        }
        unsigned best_inserted_index;
        if (_locks_enabled)
          best_inserted_index =
              ANNIndex<T>::process_candidates_into_best_candidates_pool(
                  node_coords, des, best_L_nodes, Lsize, l, inserted_into_pool,
                  cmps, labels_to_filter_by);
        else
          best_inserted_index =
              ANNIndex<T>::process_candidates_into_best_candidates_pool(
                  node_coords, _out_nbrs[n], best_L_nodes, Lsize, l,
                  inserted_into_pool, cmps, labels_to_filter_by);

        if (best_inserted_index <= k)
          k = best_inserted_index;
        else
          ++k;
      } else
        k++;
    }
    if (stats != nullptr) {
      stats->n_hops += hops;
      stats->n_cmps += cmps;
    }
    return l;
  }

  template<typename T>
  _u32 GraphIndex<T>::greedy_search_to_fixed_point(
      const T *node_coords, const unsigned Lsize,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor> &      expanded_nodes_info,
      tsl::robin_set<unsigned> &   expanded_nodes_ids,
      std::vector<Neighbor> &best_L_nodes, QueryStats *stats) {
    std::vector<label> tmp_labels;
    return greedy_search_to_fixed_point(node_coords, Lsize, init_ids, expanded_nodes_info, expanded_nodes_ids, best_L_nodes, tmp_labels, stats);
      }


  template<typename T>
  void GraphIndex<T>::prune_candidates_alpha_rng(
      const unsigned point_id, std::vector<Neighbor> &candidate_list,
      const Parameters &parameter, std::vector<unsigned> &pruned_list) {
    unsigned degree_bound = parameter.Get<_u32>("R");
    unsigned maxc = parameter.Get<_u32>("C");
    float    alpha = parameter.Get<float>("alpha");

    if (candidate_list.size() == 0)
      return;

    // sort the candidate_list based on distance to query
    std::sort(candidate_list.begin(), candidate_list.end());

    std::vector<Neighbor> result;
    result.reserve(degree_bound);

    auto               pool_size = (_u32) candidate_list.size();
    std::vector<float> occlude_factor(pool_size, 0);

    unsigned start = 0;
    while (result.size() < degree_bound && (start) < candidate_list.size() &&
           start < maxc) {
      auto &p = candidate_list[start];
      if (p.id == point_id) {
        start++;
        continue;
      }
      if (occlude_factor[start] > alpha) {
        start++;
        continue;
      }
      occlude_factor[start] = std::numeric_limits<float>::max();
      result.push_back(p);
      for (unsigned t = start + 1; t < candidate_list.size() && t < maxc; t++) {
        if (occlude_factor[t] > alpha)
          continue;

        bool prune_allowed = true;
        if (this->_filtered_index) {
            _u32 a = p.id;
            _u32 b = candidate_list[t].id;
            for (auto &x : this->_pts_to_labels[b]) {
              if (std::find(this->_pts_to_labels[a].begin(), this->_pts_to_labels[a].end(),
                            x) == this->_pts_to_labels[a].end()) {
                prune_allowed = false;
              }
              if (!prune_allowed)
                break;
            }
        }
        if (!prune_allowed)
          continue;



        float djk = this->_distance->compare(
            this->_data + this->_aligned_dim * (_u64) candidate_list[t].id,
            this->_data + this->_aligned_dim * (_u64) p.id,
            (unsigned) this->_aligned_dim);
        occlude_factor[t] =
            (std::max)(occlude_factor[t], candidate_list[t].distance / djk);
      }
      start++;
    }

    /* Add all the nodes in result into a variable called pruned_list
     * So this contains all the neighbors of id point_id
     */
    pruned_list.clear();
    assert(result.size() <= degree_bound);
    for (auto iter : result) {
      if (iter.id != point_id)
        pruned_list.emplace_back(iter.id);
    }
  }

  template<typename T>
  void GraphIndex<T>::prune_candidates_top_K(
      const unsigned point_id, std::vector<Neighbor> &candidate_list,
      const Parameters &parameter, std::vector<unsigned> &pruned_list) {
    unsigned degree_bound = parameter.Get<unsigned>("R");

    if (candidate_list.size() == 0)
      return;

    // sort the candidate_list based on distance to query
    std::sort(candidate_list.begin(), candidate_list.end());

    pruned_list.clear();

    unsigned start = 0;
    while (pruned_list.size() < degree_bound &&
           (start) < candidate_list.size()) {
      auto &p = candidate_list[start];
      if (p.id == point_id) {
        start++;
        continue;
      }
      pruned_list.push_back(p.id);
      start++;
    }
  }

  template<typename T>
  void GraphIndex<T>::add_reciprocal_edges(unsigned               n,
                                           std::vector<unsigned> &pruned_list,
                                           const Parameters &     parameters) {
    const auto degree_bound = parameters.Get<unsigned>("R");

    const auto prune_rule = parameters.Get<unsigned>("pruning_rule");

    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      /* des_pool contains the neighbors of the neighbors of n */

      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        WriteLock guard(this->_locks[des]);
      auto &                des_pool = this->_out_nbrs[des];
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

        _u64 reserveSize =
            (_u64)(std::ceil(1.05 * VAMANA_SLACK_FACTOR * degree_bound));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != des) {
            float dist = this->_distance->compare(
                this->_data + this->_aligned_dim * (_u64) des,
                this->_data + this->_aligned_dim * (_u64) cur_nbr,
                (unsigned) this->_aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        if (prune_rule == 0)
          this->prune_candidates_alpha_rng(des, dummy_pool, parameters,
                                           new_out_neighbors);
        else
          this->prune_candidates_top_K(des, dummy_pool, parameters,
                                       new_out_neighbors);
        {
          WriteLock guard(this->_locks[des]);
          this->_out_nbrs[des].clear();
          for (auto new_nbr : new_out_neighbors) {
            this->_out_nbrs[des].emplace_back(new_nbr);
          }
        }
      }
    }
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

    std::cout << std::setw(16) << "Percentile" << std::setw(16)
                << "Out Degree" << std::setw(16) << "In Degree" << std::endl;
    std::cout << "======================================================="
                << std::endl;
    for (_u32 p = 0; p < 100; p += 10) {
      std::cout << std::setw(16) << p << std::setw(16)
                  << out_degrees[(_u64)((p / 100.0) * this->_num_points)]
                  << std::setw(16)
                  << in_degrees[(_u64)((p / 100.0) * this->_num_points)].second
                  << std::endl;
    }
    std::cout << std::setw(16) << "100" << std::setw(16)
                << out_degrees[this->_num_points - 1] << std::setw(16)
                << in_degrees[this->_num_points - 1].second << std::endl;

    std::cout << std::setprecision(3)
                << (100.0 * unreachable_count) / this->_num_points
                << "\% points are unreachable and " << std::flush;
    std::cout << in_degrees[this->_num_points - 1].first
                << " is the most popular in-degree node." << std::endl;
  }

  // EXPORTS
  template class GraphIndex<float>;
  template class GraphIndex<int8_t>;
  template class GraphIndex<uint8_t>;
}  // namespace grann
