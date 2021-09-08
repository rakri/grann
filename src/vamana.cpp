// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "vamana.h"


namespace grann {

  // Initialize an vamana with metric m, load the data of type T with filename
  // (bin), and initialize num_points
  template<typename T>
  Vamana<T>::Vamana(Metric m, const char *filename, std::vector<_u32> &list_of_ids)
      : GraphIndex<T>(m,filename,list_of_ids) {
        grann::cout<<"Initialized Vamana Object with " << this->_num_points << "points, dim=" << this->_dim << std::endl;
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
    for (unsigned i = 0; i < this->_num_points + this->_num_steiner_pts; i++) {
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
                  << ((float) total_gr_edges) /
                         ((float) (this->_num_points + this->_num_steiner_pts))
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
                    << _out_nbrs.size() << " points and loaded dataset has "
                    << _num_points << " points. " << std::endl;
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
        center[j] += _data[i * this->_aligned_dim + j];

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
    const T *             node_coords = this->_data + this->_aligned_dim * node_id;
    std::vector<Neighbor> best_L_nodes;

    if (init_ids.size() == 0)
      init_ids.emplace_back(this->_start_node);

    greedy_search_to_fixed_point(node_coords, l_build, init_ids, expanded_nodes_info,
                           expanded_nodes_ids, best_L_nodes);
  }

  template<typename T>
  void Vamana<T>::occlude_list(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result) {
    auto               pool_size = (_u32) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    occlude_list(pool, alpha, degree, maxc, result, occlude_factor);
  }

  template<typename T, typename TagT>
  void Vamana<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result,
                                    std::vector<float> &   occlude_factor) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      unsigned start = 0;
      float    eps =
          cur_alpha +
          0.01;  // used for MIPS, where we store a value of eps in cur_alpha to
                 // denote pruned out entries which we can skip in later rounds.
      while (result.size() < degree && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
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
              this->_data + this->_aligned_dim * (size_t) p.id, (unsigned) this->_aligned_dim);
            occlude_factor[t] =
                (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T>
  void Vamana<T>::prune_neighbors(const unsigned         location,
                                       std::vector<Neighbor> &pool,
                                       const Parameters &     parameter,
                                       std::vector<unsigned> &pruned_list) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    float    alpha = parameter.Get<float>("alpha");

    if (pool.size() == 0)
      return;

    //_max_degree = (std::max)(this->_max_degree, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, alpha, range, maxc, result, occlude_factor);

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
    }
                                       }

  /* batch_inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T>
  void Vamana<T>::batch_inter_insert(
      unsigned n, const std::vector<unsigned> &pruned_list,
      const Parameters &parameter, std::vector<unsigned> &need_to_sync) {
    const auto range = parameter.Get<unsigned>("R");

    // assert(!src_pool.empty());

    for (auto des : pruned_list) {
      if (des == n)
        continue;
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < this->_num_points);
      if (des > this->_num_points)
        grann::cout << "error. " << des << " exceeds max_pts" << std::endl;
      /* des_pool contains the neighbors of the neighbors of n */

      {
        LockGuard guard(this->_locks[des]);
        if (std::find(this->_out_nbrs[des].begin(), this->_out_nbrs[des].end(), n) ==
            this->_out_nbrs[des].end()) {
          this->_out_nbrs[des].push_back(n);
          if (this->_out_nbrs[des].size() > (unsigned) (range * VAMANA_SLACK_FACTOR))
            need_to_sync[des] = 1;
        }
      }  // des lock is released by this point
    }
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T>
  void Vamana<T>::inter_insert(unsigned               n,
                                    std::vector<unsigned> &pruned_list,
                                    const Parameters &     parameter,
                                    bool                   update_in_nbrs) {
    const auto range = parameter.Get<unsigned>("R");
    assert(n >= 0 && n < _num_points);
    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      /* des_pool contains the neighbors of the neighbors of n */
      auto &                des_pool = _out_nbrs[des];
      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        LockGuard guard(this->_locks[des]);
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < VAMANA_SLACK_FACTOR * range) {
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

        size_t reserveSize = (size_t)(std::ceil(1.05 * VAMANA_SLACK_FACTOR * range));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != des) {
            float dist =
                this->_distance->compare(this->_data + this->_aligned_dim * (size_t) des,
                                   this->_data + this->_aligned_dim * (size_t) cur_nbr,
                                   (unsigned) this->_aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        prune_neighbors(des, dummy_pool, parameter, new_out_neighbors);
        {
          LockGuard guard(this->_locks[des]);
          // DELETE IN-EDGES FROM IN_GRAPH USING APPROPRIATE LOCKS
          this->_out_nbrs[des].clear();
          for (auto new_nbr : new_out_neighbors) {
            this->_out_nbrs[des].emplace_back(new_nbr);
            if (update_in_nbrs) {
              this->_in_nbrs[new_nbr].emplace_back(des);
            }
          }
        }
      }
    }
  }
  /* Link():
   * The graph creation function.
   *    The graph will be updated periodically in NUM_SYNCS batches
   */
  template<typename T>
  void Vamana<T>::link(Parameters &parameters) {
    unsigned NUM_THREADS = parameters.Get<unsigned>("num_threads");
    if (NUM_THREADS != 0)
      omp_set_num_threads(NUM_THREADS);

    uint32_t NUM_SYNCS =
        (unsigned) DIV_ROUND_UP(this->_num_points + this->_num_steiner_pts, (64 * 64));
    if (NUM_SYNCS < 40)
      NUM_SYNCS = 40;
    grann::cout << "Number of syncs: " << NUM_SYNCS << std::endl;

    if (NUM_THREADS != 0)
      omp_set_num_threads(NUM_THREADS);

    const unsigned argL = parameters.Get<unsigned>("L");  // Search list size
    const unsigned range = parameters.Get<unsigned>("R");
    const float    last_round_alpha = parameters.Get<float>("alpha");
    unsigned       L = argL;

    std::vector<unsigned> Lvec;
    Lvec.push_back(L);
    Lvec.push_back(L);
    const unsigned NUM_RNDS = 2;

    // Max degree of graph
    // Pruning parameter
    // Set alpha=1 for the first pass; use specified alpha for last pass
    parameters.Set<float>("alpha", 1);

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<unsigned> visit_order;
    visit_order.reserve(this->_num_points + this->_num_steiner_pts);
    for (unsigned i = 0; i < (unsigned) this->_num_points; i++) {
      visit_order.emplace_back(i);
    }

    for (unsigned i = 0; i < (unsigned) this->_num_steiner_pts; ++i)
      visit_order.emplace_back((unsigned) (this->_num_points + i));

    // if there are frozen points, the first such one is set to be the _start_node
    if (this->_num_steiner_pts > 0)
      this->_start_node = (unsigned) this->_num_points;
    else
      this->_start_node = calculate_entry_point();

    this->_out_nbrs.reserve(_num_points + _num_steiner_pts);
    this->_out_nbrs.resize(_num_points + _num_steiner_pts);

    for (uint64_t p = 0; p < this->_num_points + this->_num_steiner_pts; p++) {
      this->_out_nbrs[p].reserve((size_t)(std::ceil(range * VAMANA_SLACK_FACTOR * 1.05)));
    }

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // creating a initial list to begin the search process. it has _start_node and
    // random other nodes
    std::set<unsigned> unique_start_points;
    unique_start_points.insert(this->_start_node);

    std::vector<unsigned> init_ids;
    for (auto pt : unique_start_points)
      init_ids.emplace_back(pt);

    grann::Timer link_timer;
    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      L = Lvec[rnd_no];

      if (rnd_no == NUM_RNDS - 1) {
        if (last_round_alpha > 1)
          parameters.Set<float>("alpha", last_round_alpha);
      }

      double   sync_time = 0, total_sync_time = 0;
      double   inter_time = 0, total_inter_time = 0;
      size_t   inter_count = 0, total_inter_count = 0;
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(_num_points, NUM_SYNCS);  // size of each batch
      std::vector<unsigned> need_to_sync(this->_num_points + this->_num_steiner_pts, 0);

      std::vector<std::vector<unsigned>> pruned_list_vector(round_size);

      for (uint32_t sync_num = 0; sync_num < NUM_SYNCS; sync_num++) {
        size_t start_id = sync_num * round_size;
        size_t end_id =
            (std::min)(this->_num_points + this->_num_steiner_pts, (sync_num + 1) * round_size);

        auto s = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff;

#pragma omp parallel for schedule(dynamic)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          auto                     node = visit_order[node_ctr];
          size_t                   node_offset = node_ctr - start_id;
          tsl::robin_set<unsigned> visited;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          // get nearest neighbors of n in tmp. pool contains all the
          // points that were checked along with their distance from
          // n. visited contains all the points visited, just the ids
          std::vector<Neighbor> pool;
          pool.reserve(L * 2);
          visited.reserve(L * 2);
          get_expanded_nodes(node, L, init_ids, pool, visited);
          /* check the neighbors of the query that are not part of
           * visited, check their distance to the query, and add it to
           * pool.
           */
          if (!this->_out_nbrs[node].empty())
            for (auto id : this->_out_nbrs[node]) {
              if (visited.find(id) == visited.end() && id != node) {
                float dist =
                    this->_distance->compare(this->_data + this->_aligned_dim * (size_t) node,
                                       this->_data + this->_aligned_dim * (size_t) id,
                                       (unsigned) this->_aligned_dim);
                pool.emplace_back(Neighbor(id, dist, true));
                visited.insert(id);
              }
            }
          prune_neighbors(node, pool, parameters, pruned_list);
        }
        diff = std::chrono::high_resolution_clock::now() - s;
        sync_time += diff.count();

// prune_neighbors will check pool, and remove some of the points and
// create a cut_graph, which contains neighbors for point n
#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          _u64                   node = visit_order[node_ctr];
          size_t                 node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          _out_nbrs[node].clear();
          for (auto id : pruned_list)
            _out_nbrs[node].emplace_back(id);
        }
        s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = start_id; node_ctr < (_s64) end_id; ++node_ctr) {
          auto                   node = visit_order[node_ctr];
          _u64                   node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          batch_inter_insert(node, pruned_list, parameters, need_to_sync);
          //          inter_insert(node, pruned_list, parameters, 0);
          pruned_list.clear();
          pruned_list.shrink_to_fit();
        }

#pragma omp parallel for schedule(dynamic, 65536)
        for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size());
             node_ctr++) {
          auto node = visit_order[node_ctr];
          if (need_to_sync[node] != 0) {
            need_to_sync[node] = 0;
            inter_count++;
            tsl::robin_set<unsigned> dummy_visited(0);
            std::vector<Neighbor>    dummy_pool(0);
            std::vector<unsigned>    new_out_neighbors;

            for (auto cur_nbr : _out_nbrs[node]) {
              if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                  cur_nbr != node) {
                float dist =
                    _distance->compare(_data + _aligned_dim * (size_t) node,
                                       _data + _aligned_dim * (size_t) cur_nbr,
                                       (unsigned) _aligned_dim);
                dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                dummy_visited.insert(cur_nbr);
              }
            }
            prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

            _out_nbrs[node].clear();
            for (auto id : new_out_neighbors)
              _out_nbrs[node].emplace_back(id);
          }
        }

        diff = std::chrono::high_resolution_clock::now() - s;
        inter_time += diff.count();

        if ((sync_num * 100) / NUM_SYNCS > progress_counter) {
          grann::cout.precision(4);
          grann::cout << "Completed  (round: " << rnd_no
                        << ", sync: " << sync_num << "/" << NUM_SYNCS
                        << " with L " << L << ")"
                        << " sync_time: " << sync_time << "s"
                        << "; inter_time: " << inter_time << "s" << std::endl;

          total_sync_time += sync_time;
          total_inter_time += inter_time;
          total_inter_count += inter_count;
          sync_time = 0;
          inter_time = 0;
          inter_count = 0;
          progress_counter += 5;
        }
      }
// Gopal. Splittng grann_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#ifdef GRANN_BUILD
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
      grann::cout << "Completed Pass " << rnd_no << " of data using L=" << L
                    << " and alpha=" << parameters.Get<float>("alpha")
                    << ". Stats: ";
      grann::cout << "search+prune_time=" << total_sync_time
                    << "s, inter_time=" << total_inter_time
                    << "s, inter_count=" << total_inter_count << std::endl;
    }

    grann::cout << "Starting final cleanup.." << std::flush;
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size()); node_ctr++) {
      auto node = visit_order[node_ctr];
      if (_out_nbrs[node].size() > range) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);
        std::vector<unsigned>    new_out_neighbors;

        for (auto cur_nbr : _out_nbrs[node]) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != node) {
            float dist =
                _distance->compare(_data + _aligned_dim * (size_t) node,
                                   _data + _aligned_dim * (size_t) cur_nbr,
                                   (unsigned) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        prune_neighbors(node, dummy_pool, parameters, new_out_neighbors);

        _out_nbrs[node].clear();
        for (auto id : new_out_neighbors)
          _out_nbrs[node].emplace_back(id);
      }
    }
    grann::cout << "done. Link time: "
                  << ((double) link_timer.elapsed() / (double) 1000000) << "s"
                  << std::endl;
  }

  template<typename T>
  void Vamana<T>::build(Parameters &             parameters,
                             const std::vector<TagT> &tags) {
    if (_enable_tags) {
      if (tags.size() != _num_points) {
        std::cerr << "#Tags should be equal to #points" << std::endl;
        throw grann::ANNException("#Tags must be equal to #points", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
      }
      for (size_t i = 0; i < tags.size(); ++i) {
        _tag_to_location[tags[i]] = (unsigned) i;
        _location_to_tag[(unsigned) i] = tags[i];
      }
    }
    grann::cout << "Starting vamana build..." << std::endl;
    link(parameters);  // Primary func for creating graph

    if (_support_eager_delete) {
      update_in_nbrs();  // copying values to in_graph
    }

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _num_points; i++) {
      auto &pool = _out_nbrs[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    grann::cout << "Degree: max:" << max
                  << "  avg:" << (float) total / (float) _num_points << "  min:" << min
                  << "  count(deg<2):" << cnt << "\n"
                  << "Vamana built." << std::endl;
    _max_degree = (std::max)((unsigned) max, _max_degree);
    _has_built = true;
  }

  template<typename T>
  std::pair<uint32_t, uint32_t> Vamana<T>::search(const T *      query,
                                                       const size_t   K,
                                                       const unsigned L,
                                                       unsigned *     indices) {
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * L);
    std::vector<Neighbor>    best_L_nodes, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_start_node);
    }
    auto retval =
        greedy_search_to_fixed_point(query, L, init_ids, expanded_nodes_info,
                               expanded_nodes_ids, best_L_nodes);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      indices[pos] = it.id;
      pos++;
      if (pos == K)
        break;
    }
    return retval;
  }


  // EXPORTS
  template GRANN_DLLEXPORT class Vamana<float>;
  template GRANN_DLLEXPORT class Vamana<int8_t>;
  template GRANN_DLLEXPORT class Vamana<uint8_t>;
}  // namespace grann
