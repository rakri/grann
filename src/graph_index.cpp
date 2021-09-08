// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"
#include "graph_index.h"


namespace grann {

  // Initialize a generic graph-based index with metric m, load the data of type T with filename
  // (bin)
  template<typename T>
  GraphIndex<T>::GraphIndex(Metric m, const char *filename, std::vector<_u32> &list_of_ids) : ANNIndex<T>(m, filename, list_of_ids)
       { // Graph Index class constructor loads the data and sets num_points, dim, etc.
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
  void GraphIndex<T>::greedy_search_to_fixed_point(
      const T *node_coords, const unsigned Lsize,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor> &      expanded_nodes_info,
      tsl::robin_set<unsigned> &   expanded_nodes_ids,
      std::vector<Neighbor> &      best_L_nodes, QueryStats *stats) {
    best_L_nodes.resize(Lsize + 1);
    expanded_nodes_info.reserve(10 * Lsize);
    expanded_nodes_ids.reserve(10 * Lsize);

    unsigned                 l = 0;
    Neighbor                 nn;
    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(Lsize * 20);

    for (auto id : init_ids) {
      nn = Neighbor(id,
                    this->_distance->compare(this->_data + this->_aligned_dim * (size_t) id,
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
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        expanded_nodes_info.emplace_back(best_L_nodes[k]);
        expanded_nodes_ids.insert(n);

        for (unsigned m = 0; m < _out_nbrs[n].size(); ++m) {
          unsigned id = _out_nbrs[n][m];
          if (inserted_into_pool.find(id) == inserted_into_pool.end()) {
            inserted_into_pool.insert(id);

            if ((m + 1) < _out_nbrs[n].size()) {
              auto nextn = _out_nbrs[n][m + 1];
              grann::prefetch_vector(
                  (const char *) this->_data + this->_aligned_dim * (size_t) nextn,
                  sizeof(T) * this->_aligned_dim);
            }

            cmps++;
            float dist = this->_distance->compare(node_coords,
                                            this->_data + this->_aligned_dim * (size_t) id,
                                            (unsigned) this->_aligned_dim);

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else
        k++;
    }
    if (stats != nullptr) {
      stats->n_hops = hops;
      stats->n_cmps = cmps;
    }
  }


  // EXPORTS
  template GRANN_DLLEXPORT class GraphIndex<float>;
  template GRANN_DLLEXPORT class GraphIndex<int8_t>;
  template GRANN_DLLEXPORT class GraphIndex<uint8_t>;
}  // namespace grann
