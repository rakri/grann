// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <unordered_set>
#include "utils.h"
#include "vamana.h"

namespace grann {

  // Initialize an vamana with metric m, load the data of type T with filename
  // (bin), and initialize num_points
  template<typename T>
  Vamana<T>::Vamana(Metric m, const char *filename,
                    std::vector<_u32> &list_of_tags,
                        std::string        labels_fname)
      : GraphIndex<T>(m, filename, list_of_tags, labels_fname) {
    std::cout << "Initialized Vamana Object with " << this->_num_points
                << " points, dim=" << this->_dim << "." << std::endl;
  }

  template<typename T>
  Vamana<T>::Vamana(Metric m) : GraphIndex<T>(m) {
    std::cout << "Initialized Empty Vamana Object." << std::endl;
  }

  // save the graph vamana on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T>
  void Vamana<T>::save(const char *filename) {
    reorder(filename, 11, 72);
    ANNIndex<T>::save_data_and_tags_and_labels(filename);

    if (this->_filtered_index) {
      if (this->_filter_to_medoid_id.size()) {
        std::ofstream medoid_writer(std::string(filename) +
                                    "_labels_to_medoids.txt");
        for (auto iter : this->_filter_to_medoid_id) {
          medoid_writer << iter.first << ", " << iter.second << std::endl;
          std::cout << iter.first << ", " << iter.second << std::endl;
        }
        medoid_writer.close();  
    }
    }

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
    ANNIndex<T>::load_data_and_tags_and_labels(filename);

      std::string labels_to_medoids_file = std::string(filename) + "_labels_to_medoids.txt";
      if (file_exists(labels_to_medoids_file)) {
        std::ifstream medoid_stream(labels_to_medoids_file);

        std::string line, token;
        unsigned    line_cnt = 0;

        _filter_to_medoid_id.clear();

        while (std::getline(medoid_stream, line)) {
          std::istringstream iss(line);
          _u32               cnt = 0;
          _u32               medoid = 0;
          std::string        label;
          while (std::getline(iss, token, ',')) {
            token.erase(std::remove(token.begin(), token.end(), '\n'),
                        token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'),
                        token.end());
            if (cnt == 0)
              label = token;
            else
              medoid = (_u32) stoul(token);
            cnt++;
          }
          _filter_to_medoid_id[label] = medoid;
          line_cnt++;
        }
      }


    std::ifstream in(filename, std::ios::binary);
    _u64          expected_file_size;
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &this->_max_degree, sizeof(unsigned));
    in.read((char *) &this->_start_node, sizeof(unsigned));
    std::cout << "Loading vamana index " << filename << "..." << std::flush;

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
        std::cout << "." << std::flush;
    }
    if (this->_out_nbrs.size() != this->_num_points) {
      std::cout << "ERROR. mismatch in number of points. Graph has "
                  << this->_out_nbrs.size() << " points and loaded dataset has "
                  << this->_num_points << " points. " << std::endl;
      return;
    }

    std::cout << "..done. Vamana has " << nodes << " nodes and " << cc
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
      tsl::robin_set<unsigned> &expanded_nodes_ids, const std::vector<label> &labels_to_accept) {
    const T *node_coords = this->_data + this->_aligned_dim * node_id;
    std::vector<Neighbor> best_L_nodes;

    if (init_ids.size() == 0)
      init_ids.emplace_back(this->_start_node);

    this->greedy_search_to_fixed_point(node_coords, l_build, init_ids,
                                       expanded_nodes_info, expanded_nodes_ids,
                                       best_L_nodes, labels_to_accept);
  }

  template<typename T>
  void Vamana<T>::calculate_label_specific_medoids() {
    if (this->_filtered_index == false) {
      return;
    }
    std::cout<<"Processing label-specific medoids" << std::endl;
   _u32 counter = 0;
#pragma omp parallel for schedule(dynamic, 1)
    for (uint32_t lbl = 0; lbl < this->_labels.size(); lbl++) {
      auto itr = this->_labels.begin();
      std::advance(itr, lbl);
      auto &x = *itr;
      //      if (x == _universal_label) {   // we are not storing medoid for
      //      universal label
      //        continue;
      //      }
      std::vector<_u32> filtered_points;
      for (_u32 i = 0; i < this->_num_points; i++) {
        if (std::find(this->_pts_to_labels[i].begin(), this->_pts_to_labels[i].end(), x) !=
                this->_pts_to_labels[i].end() ||
            (this->_use_universal_label &&
             (std::find(this->_pts_to_labels[i].begin(), this->_pts_to_labels[i].end(),
                        this->_universal_label) != this->_pts_to_labels[i].end())))
          filtered_points.emplace_back(i);
      }
      if (filtered_points.size() != 0) {
#pragma omp critical
        {
          _u32 num_cands = 25;
          _u32 best_medoid = 0;
          _u32 best_medoid_count = std::numeric_limits<_u32>::max();
          for (_u32 cnd = 0; cnd < num_cands; cnd++) {
            _u32 cur_cnd = filtered_points[rand() % filtered_points.size()];
            _u32 cur_cnt = std::numeric_limits<_u32>::max();
            if (this->_medoid_counts.find(cur_cnd) == this->_medoid_counts.end()) {
              this->_medoid_counts[cur_cnd] = 0;
              cur_cnt = 0;
            } else {
              cur_cnt = this->_medoid_counts[cur_cnd];
            }
            if (cur_cnt < best_medoid_count || cnd == 0) {
              best_medoid_count = cur_cnt;
              best_medoid = cur_cnd;
            }
          }

          this->_filter_to_medoid_id[x] = best_medoid;
          this->_medoid_counts[best_medoid]++;
          std::stringstream a;
          a << "Medoid of " << x << " is " << best_medoid << std::endl;
          std::cout << a.str();
        }
      }
#pragma omp critical
      counter++;
      std::stringstream a;
      a << ((100.0 * counter) / this->_labels.size()) << "\% processed \r";
      std::cout << a.str() << std::flush;
    }
  }

  template<typename T>
  void Vamana<T>::build(const Parameters &build_parameters) {
    grann::Timer build_timer;

    if (this->_filtered_index) {
      calculate_label_specific_medoids();
    }

    unsigned num_threads = build_parameters.Get<unsigned>("num_threads");
    unsigned L = build_parameters.Get<unsigned>("L");
    unsigned degree_bound = build_parameters.Get<unsigned>("R");
    float    alpha = build_parameters.Get<float>("alpha");

    std::cout << "Starting vamana build with listSize L=" << L
                << ", degree bound R=" << degree_bound
                << ", and alpha=" << alpha << std::endl;

    this->_locks_enabled =
        true;  // we dont need locks for pure search on a pre-built index
    this->_locks = std::vector<std::shared_timed_mutex>(this->_num_points);
    this->_out_nbrs.resize(this->_num_points);
    for (auto &x : this->_out_nbrs)
      x.reserve(1.05 * VAMANA_SLACK_FACTOR * degree_bound);

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    this->_start_node = ANNIndex<T>::calculate_medoid_of_data();
    std::cout << "Medoid identified as " << this->_start_node << std::endl;

    _u32             progress_milestone = (_u32)(this->_num_points / 10);
    std::atomic<int> milestone_marker{0};

#pragma omp parallel for schedule(static, 64)
    for (_u32 location = 0; location < this->_num_points; location++) {
      if (location % progress_milestone == 0) {

        std::stringstream msg;
        _u32 a = milestone_marker;
        msg << "\r" << (a * 10) << "\% of build completed...";
        std::cout << msg.str();
        ++milestone_marker;
      }

      std::vector<Neighbor> pool;
      std::vector<Neighbor> tmp;
      tsl::robin_set<_u32>  visited;
      pool.reserve(2 * L);
      tmp.reserve(2 * L);
      visited.reserve(20 * L);

      std::vector<_u32> pruned_list;
      std::vector<_u32> init_ids;
      
                if (!this->_filtered_index)
            get_expanded_nodes(location, L, init_ids, pool, visited);
          else {
            std::vector<_u32> filter_specific_start_nodes;
            for (auto &x : this->_pts_to_labels[location]) {
              if (_filter_to_medoid_id.find(x) == _filter_to_medoid_id.end()) {
                  continue;
              }
              filter_specific_start_nodes.emplace_back(_filter_to_medoid_id[x]);
            }
            get_expanded_nodes(location, L, filter_specific_start_nodes, pool,
                               visited, this->_pts_to_labels[location]);
          }

//      get_expanded_nodes(location, L, init_ids, pool, visited);

      this->prune_candidates_alpha_rng(location, pool, build_parameters,
                                       pruned_list);

      this->_out_nbrs[location].reserve(
          (_u64)(VAMANA_SLACK_FACTOR * degree_bound));
      {
        WriteLock guard(this->_locks[location]);
        for (auto link : pruned_list)
          this->_out_nbrs[location].emplace_back(link);
      }
      GraphIndex<T>::add_reciprocal_edges(
          location, pruned_list,
          build_parameters);  // add reverse edges
    }
    std::cout << "\nStarting final cleanup.." << std::flush;
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

    std::cout << "done." << std::endl;
    this->_has_built = true;
    this->update_degree_stats();

    std::cout << "Total build time: "
                << ((double) build_timer.elapsed() / (double) 1000000) << "s"
                << std::endl;
  }


  template<typename T>
  void Vamana<T>::partition_packing(
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
  void Vamana<T>::reorder(
      const std::string filename, const unsigned omega, const unsigned threads) {
    std::vector<unsigned>        p_order(this->_num_points);
    std::vector<unsigned>        o_order(this->_num_points);
    std::unordered_set<unsigned> initial;
    boost::dynamic_bitset<>      deleted{this->_num_points, 0};


    this->_in_nbrs.reserve(this->_num_points);
    this->_in_nbrs.resize(this->_num_points);
    for (unsigned i = 0; i < this->_in_nbrs.size(); i++) {
      this->_in_nbrs[i].clear();
    }
    for (unsigned i = 0; i < this->_out_nbrs.size(); i++) {
      for (unsigned j = 0; j < this->_out_nbrs[i].size(); j++) {
        this->_in_nbrs[this->_out_nbrs[i][j]].emplace_back(i);
      }
    }



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
  _u32 Vamana<T>::search(const T *query, _u32 res_count,
                         const Parameters &search_params, _u32 *indices,
                         float *distances, QueryStats *stats,
												 std::vector<label> search_filters) {
    _u32                     search_list_size = search_params.Get<_u32>("L");
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * search_list_size);
    std::vector<Neighbor>    top_candidate_list, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (search_filters.size()!=0) {
      for (auto &x : search_filters)
    if (_filter_to_medoid_id.find(x) != _filter_to_medoid_id.end())
      init_ids.emplace_back(_filter_to_medoid_id[x]);
    } else
    init_ids.emplace_back(this->_start_node);

    auto algo_fetched_count = this->greedy_search_to_fixed_point(
        query, search_list_size, init_ids, expanded_nodes_info,
        expanded_nodes_ids, top_candidate_list, search_filters, stats);

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
