// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "graph_index.h"
#include <iomanip>
#include <sstream>
#include <unordered_set>

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
  void GraphIndex<T>::partition_packing(
      unsigned *p_order, const unsigned seed_node, const unsigned omega,
      std::unordered_set<unsigned> &initial, boost::dynamic_bitset<> &deleted) {
    std::unordered_map<unsigned, unsigned> counts;

    p_order[0] = seed_node;

    for (unsigned i = 1; i < omega; i++) {
      unsigned ve = p_order[i - 1];
//      auto     visit_in = visit_counts_in.find(ve);
//      auto     visit_out = visit_counts_out.find(ve);
      for (unsigned j = 0; j < this->_out_nbrs[ve].size(); j++) {
        if (deleted[this->_out_nbrs[ve][j]] == false) {
          if (counts.find(this->_out_nbrs[ve][j]) == counts.end()) {
            counts[this->_out_nbrs[ve][j]] = 0;
          }
          counts[this->_out_nbrs[ve][j]] += 1;
//          if (visit_out != visit_counts_out.end()) {
//            counts[this->_out_nbrs[ve][j]] += visit_out->second[j] * 3;
//          }
        }
      }
      for (unsigned j = 0; j < this->_in_nbrs[ve].size(); j++) {
        if (deleted[this->_in_nbrs[ve][j]] == false) {
          if (counts.find(this->_in_nbrs[ve][j]) == counts.end()) {
            counts[this->_in_nbrs[ve][j]] = 0;
          }
          counts[this->_in_nbrs[ve][j]] += 1;
//          if (visit_in != visit_counts_in.end()) {
//            counts[this->_in_nbrs[ve][j]] += visit_in->second[j] * 3;
//          }
        }
        for (unsigned k = 0; k < this->_out_nbrs[this->_in_nbrs[ve][j]].size(); k++) {
          if (deleted[this->_out_nbrs[this->_in_nbrs[ve][j]][k]] == false) {
            if (counts.find(this->_out_nbrs[this->_in_nbrs[ve][j]][k]) ==
                counts.end()) {
              counts[this->_out_nbrs[this->_in_nbrs[ve][j]][k]] = 0;
            }
            counts[this->_out_nbrs[this->_in_nbrs[ve][j]][k]] += 1;
          }
        }
      }

      bool found = false;
      while (counts.size() > 0) {
        auto     max_it = counts.begin();
        unsigned max_val = max_it->second;
        for (auto itr = counts.begin(); itr != counts.end(); itr++) {
          if (itr->second > max_val) {
            max_it = itr;
            max_val = itr->second;
          }
        }
#pragma omp critical
        {
          if (deleted[max_it->first] == false) {
            deleted[max_it->first] = true;
            initial.erase(max_it->first);
            found = true;
          } else {
            counts.erase(max_it->first);
          }
        }
        if (found) {
          p_order[i] = max_it->first;
          counts.erase(p_order[i]);
          break;
        }
      }
      while (!found) {
        for (auto itr = initial.begin(); itr != initial.end(); itr++) {
          if (deleted[*itr] == false) {
            p_order[i] = *(itr);
            break;
          }
        }
#pragma omp critical
        {
          if (deleted[p_order[i]] == false) {
            deleted[p_order[i]] = true;
            initial.erase(p_order[i]);
            found = true;
          }
        }

        if (found) {
          counts.erase(p_order[i]);
          break;
        }
      }
    }
  }


  template<typename T>
  void GraphIndex<T>::greedy_ordering(const std::string filename,
                                       const unsigned omega) {
    std::vector<unsigned>        p_order(this->_num_points);
    std::vector<unsigned>        o_order(this->_num_points);
    std::vector<unsigned>        counts(this->_num_points, 0);
    std::unordered_set<unsigned> deleted;
    std::map<unsigned, std::unordered_set<unsigned>, std::greater<unsigned>>
        value_keys;

    _in_nbrs.reserve(this->_num_points);
    _in_nbrs.resize(this->_num_points);
    for (unsigned i = 0; i < _in_nbrs.size(); i++) {
      _in_nbrs[i].clear();
    }

    std::vector<std::mutex> in_locks (this->_num_points);

#pragma omp parallel for schedule(dynamic, 128)    
    for (unsigned i = 0; i < _out_nbrs.size(); i++) {
      for (unsigned j = 0; j < _out_nbrs[i].size(); j++) {
/*        if (std::find(_in_nbrs[_out_nbrs[i][j]].begin(),
                      _in_nbrs[_out_nbrs[i][j]].end(),
                      i) != _in_nbrs[_out_nbrs[i][j]].end()) {
          std::cout << "Duplicates found" << std::endl;
        } */
        {
          grann::LockGuard lock(in_locks[_out_nbrs[i][j]]);
        _in_nbrs[_out_nbrs[i][j]].emplace_back(i);
        }
      }
    }
    std::cout<<"In-graph computed. Now going to work on ordering.\n" << std::endl;
    _u32 start_node = 0;
    p_order[0] = start_node;
    deleted.insert(start_node);
    std::unordered_set<unsigned> initial;
    for (unsigned i = 0; i < this->_num_points; i++) {
      if (i != start_node) {
        initial.insert(i);
      }
    }
    value_keys.insert(
        std::pair<unsigned, std::unordered_set<unsigned>>(0, initial));

    grann::Timer reorder_timer;
    for (unsigned i = 1; i < this->_num_points; i++) {
      if (i % 100 == 0 && i>= 1) {
      double elapsed_secs = (reorder_timer.elapsed())/(1000000);
      double estimated_time_left = ((double)this->_num_points/ (double)i) * elapsed_secs;
      std::stringstream a;
      a << "\r" << ((100.0 * i) / this->_num_points) << "\% processed. Estimated time left: " << estimated_time_left - elapsed_secs<< "s";
      std::cout << a.str() << std::flush;
      }
      unsigned ve = p_order[i - 1];
      for (unsigned j = 0; j < _out_nbrs[ve].size(); j++) {
        if (deleted.find(_out_nbrs[ve][j]) == deleted.end()) {
          counts[_out_nbrs[ve][j]] += 1;
          auto it = value_keys.find(counts[_out_nbrs[ve][j]]);
          if (it == value_keys.end()) {
            std::unordered_set<unsigned> local;
            local.insert(_out_nbrs[ve][j]);
            value_keys.insert(std::pair<unsigned, std::unordered_set<unsigned>>(
                counts[_out_nbrs[ve][j]], local));
          } else {
            it->second.insert(_out_nbrs[ve][j]);
          }
          it = value_keys.find(counts[_out_nbrs[ve][j]] - 1);
          if (it != value_keys.end()) {
            it->second.erase(_out_nbrs[ve][j]);
          }
        }
      } /*
      for (unsigned j = 0; j < _in_nbrs[ve].size(); j++) {
        if (deleted.find(_in_nbrs[ve][j]) == deleted.end()) {
          counts[_in_nbrs[ve][j]] += 1;
          auto it = value_keys.find(counts[_in_nbrs[ve][j]]);
          if (it == value_keys.end()) {
            std::unordered_set<unsigned> local;
            local.insert(_in_nbrs[ve][j]);
            value_keys.insert(std::pair<unsigned, std::unordered_set<unsigned>>(
                counts[_in_nbrs[ve][j]], local));
          } else {
            it->second.insert(_in_nbrs[ve][j]);
          }
          it = value_keys.find(counts[_in_nbrs[ve][j]] - 1);
          if (it != value_keys.end()) {
            it->second.erase(_in_nbrs[ve][j]);
          }
        }
        for (unsigned k = 0; k < _out_nbrs[_in_nbrs[ve][j]].size(); k++) {
          if (deleted.find(_out_nbrs[_in_nbrs[ve][j]][k]) ==
              deleted.end()) {
            counts[_out_nbrs[_in_nbrs[ve][j]][k]] += 1;
            auto it =
                value_keys.find(counts[_out_nbrs[_in_nbrs[ve][j]][k]]);
            if (it == value_keys.end()) {
              std::unordered_set<unsigned> local;
              local.insert(_out_nbrs[_in_nbrs[ve][j]][k]);
              value_keys.insert(
                  std::pair<unsigned, std::unordered_set<unsigned>>(
                      counts[_out_nbrs[_in_nbrs[ve][j]][k]], local));
            } else {
              it->second.insert(_out_nbrs[_in_nbrs[ve][j]][k]);
            }
            it = value_keys.find(counts[_out_nbrs[_in_nbrs[ve][j]][k]] - 1);
            if (it != value_keys.end()) {
              it->second.erase(_out_nbrs[_in_nbrs[ve][j]][k]);
            }
          }
        } 
      } */

      if (i > omega) {
        unsigned vb = p_order[i - omega - 1];
        for (unsigned j = 0; j < _out_nbrs[vb].size(); j++) {
          if (deleted.find(_out_nbrs[vb][j]) == deleted.end()) {
            // assert(counts[_out_nbrs[vb][j]] > 0);
            counts[_out_nbrs[vb][j]] -= 1;
            auto it = value_keys.find(counts[_out_nbrs[vb][j]]);
            if (it == value_keys.end()) {
              std::unordered_set<unsigned> local;
              local.insert(_out_nbrs[vb][j]);
              value_keys.insert(
                  std::pair<unsigned, std::unordered_set<unsigned>>(
                      counts[_out_nbrs[vb][j]], local));
            } else {
              it->second.insert(_out_nbrs[vb][j]);
            }
            it = value_keys.find(counts[_out_nbrs[vb][j]] + 1);
            if (it != value_keys.end()) {
              it->second.erase(_out_nbrs[vb][j]);
            }
          }
        }
        for (unsigned j = 0; j < _in_nbrs[vb].size(); j++) {
          if (deleted.find(_in_nbrs[vb][j]) == deleted.end()) {
            // assert(counts[_in_nbrs[vb][j]] > 0);
            counts[_in_nbrs[vb][j]] -= 1;
            auto it = value_keys.find(counts[_in_nbrs[vb][j]]);
            if (it == value_keys.end()) {
              std::unordered_set<unsigned> local;
              local.insert(_in_nbrs[vb][j]);
              value_keys.insert(
                  std::pair<unsigned, std::unordered_set<unsigned>>(
                      counts[_in_nbrs[vb][j]], local));
            } else {
              it->second.insert(_in_nbrs[vb][j]);
            }
            it = value_keys.find(counts[_in_nbrs[vb][j]] + 1);
            if (it != value_keys.end()) {
              it->second.erase(_in_nbrs[vb][j]);
            }
          }
          for (unsigned k = 0; k < _out_nbrs[_in_nbrs[vb][j]].size(); k++) {
            if (deleted.find(_out_nbrs[_in_nbrs[vb][j]][k]) ==
                deleted.end()) {
              // assert(counts[_out_nbrs[_in_nbrs[vb][j]][k]] > 0);
              counts[_out_nbrs[_in_nbrs[vb][j]][k]] -= 1;
              auto it =
                  value_keys.find(counts[_out_nbrs[_in_nbrs[vb][j]][k]]);
              if (it == value_keys.end()) {
                std::unordered_set<unsigned> local;
                local.insert(_out_nbrs[_in_nbrs[vb][j]][k]);
                value_keys.insert(
                    std::pair<unsigned, std::unordered_set<unsigned>>(
                        counts[_out_nbrs[_in_nbrs[vb][j]][k]], local));
              } else {
                it->second.insert(_out_nbrs[_in_nbrs[vb][j]][k]);
              }
              it = value_keys.find(counts[_out_nbrs[_in_nbrs[vb][j]][k]] +
                                   1);
              if (it != value_keys.end()) {
                it->second.erase(_out_nbrs[_in_nbrs[vb][j]][k]);
              }
            }
          }
        }
      }

      while (true) {
        auto it = value_keys.begin();
        if (it->second.size() != 0) {
          break;
        }
        value_keys.erase(it);
      }
      auto it = value_keys.begin();
      auto itr = it->second.begin();
      p_order[i] = *itr;
      it->second.erase(itr);
      counts[p_order[i]] = 0;
      deleted.insert(p_order[i]);
    }

    std::ofstream out(filename + "_loc_to_id.bin",
                      std::ios::binary | std::ios::out);
    _u32 nr = this->_num_points;
    _u32 nd = 1;
    out.write((char *) &nr, sizeof(_u32));
    out.write((char *) &nd, sizeof(_u32));    
    out.write((char *) p_order.data(), this->_num_points * sizeof(unsigned));
    out.close();

    for (unsigned i = 0; i < this->_num_points; i++) {
      o_order[p_order[i]] = i;
    }

    std::ofstream outer(filename + "_id_to_loc.bin",
                        std::ios::binary | std::ios::out);
    outer.write((char *) &nr, sizeof(_u32));
    outer.write((char *) &nd, sizeof(_u32));    
    outer.write((char *) o_order.data(), this->_num_points * sizeof(unsigned));
    outer.close();
  }



  template<typename T>
  void GraphIndex<T>::sector_reordering(
      const std::string filename, const unsigned omega, const unsigned threads) {
    std::vector<unsigned>        p_order(this->_num_points);
    std::vector<unsigned>        o_order(this->_num_points);
    std::unordered_set<unsigned> initial;
    boost::dynamic_bitset<>      deleted{this->_num_points, 0};


    this->_in_nbrs.reserve(this->_num_points);
    this->_in_nbrs.resize(this->_num_points);



    std::vector<std::mutex> in_locks (this->_num_points);

#pragma omp parallel for schedule(dynamic, 128)    
    for (unsigned i = 0; i < _out_nbrs.size(); i++) {
      for (unsigned j = 0; j < _out_nbrs[i].size(); j++) {
/*        if (std::find(_in_nbrs[_out_nbrs[i][j]].begin(),
                      _in_nbrs[_out_nbrs[i][j]].end(),
                      i) != _in_nbrs[_out_nbrs[i][j]].end()) {
          std::cout << "Duplicates found" << std::endl;
        } */
        {
          grann::LockGuard lock(in_locks[_out_nbrs[i][j]]);
        _in_nbrs[_out_nbrs[i][j]].emplace_back(i);
        }
      }
    }
    std::cout<<"In-graph computed. Now going to work on ordering.\n" << std::endl;



/*
    for (unsigned i = 0; i < this->_in_nbrs.size(); i++) {
      this->_in_nbrs[i].clear();
    }
    for (unsigned i = 0; i < this->_out_nbrs.size(); i++) {
      for (unsigned j = 0; j < this->_out_nbrs[i].size(); j++) {
        this->_in_nbrs[this->_out_nbrs[i][j]].emplace_back(i);
      }
    }
*/


    for (unsigned i = 0; i < this->_num_points; i++) {
      initial.insert(i);
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (unsigned i = 0; i < this->_num_points / omega; i++) {
      unsigned seed_node;
#pragma omp    critical
      {
        seed_node = *(initial.begin());
        deleted[seed_node] = true;
        initial.erase(initial.begin());
      }
      partition_packing(p_order.data() + i * omega, seed_node, omega, initial,
                        deleted);
    }

    if (this->_num_points % omega != 0) {
      for (unsigned i = (this->_num_points / omega) * omega; i < this->_num_points; i++) {
        p_order[i] = *(initial.begin());
        initial.erase(initial.begin());
      }
    }

    std::ofstream out(filename + "_loc_to_id.bin",
                      std::ios::binary | std::ios::out);
    _u32 nr = this->_num_points;
    _u32 nd = 1;
    out.write((char *) &nr, sizeof(_u32));
    out.write((char *) &nd, sizeof(_u32));    
    out.write((char *) p_order.data(), this->_num_points * sizeof(unsigned));
    out.close();

    for (unsigned i = 0; i < this->_num_points; i++) {
      o_order[p_order[i]] = i;
    }

    std::ofstream outer(filename + "_id_to_loc.bin",
                        std::ios::binary | std::ios::out);
    outer.write((char *) &nr, sizeof(_u32));
    outer.write((char *) &nd, sizeof(_u32));    
    outer.write((char *) o_order.data(), this->_num_points * sizeof(unsigned));
    outer.close();
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
