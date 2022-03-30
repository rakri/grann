// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "utils.h"
#include "distance.h"
#include "percentile_stats.h"
#include "index.h"

namespace grann {

  typedef std::vector<std::vector<_u32>> NeighborList;

  template<typename T>
  class GraphIndex : public ANNIndex<T> {
   public:
    GraphIndex(Metric m, const char *filename, std::vector<_u32> &list_of_tags, std::string labels_fname="");
    GraphIndex(Metric m);

    /*  Internals of the library */
   protected:
    NeighborList _out_nbrs;
    NeighborList _in_nbrs;
    _u32         _max_degree = 0;
    bool         _locks_enabled =
        false;  // will be used at build time, pure search dont need locks

    void prune_candidates_alpha_rng(const unsigned         location,
                                    std::vector<Neighbor> &pool,
                                    const Parameters &     parameter,
                                    std::vector<unsigned> &pruned_list);

    void prune_candidates_top_K(const unsigned         location,
                                std::vector<Neighbor> &pool,
                                const Parameters &     parameter,
                                std::vector<unsigned> &pruned_list);

    void add_reciprocal_edges(unsigned n, std::vector<unsigned> &pruned_list,
                              const Parameters &parameters);

    _u32 greedy_search_to_fixed_point(
        const T *node_coords, const _u32 list_size,
        const std::vector<_u32> &init_ids,
        std::vector<Neighbor> &  expanded_nodes_info,
        tsl::robin_set<_u32> &   expanded_nodes_ids,
        std::vector<Neighbor> &best_L_nodes, QueryStats *stats = nullptr);

    _u32 greedy_search_to_fixed_point(
        const T *node_coords, const _u32 list_size,
        const std::vector<_u32> &init_ids,
        std::vector<Neighbor> &  expanded_nodes_info,
        tsl::robin_set<_u32> &   expanded_nodes_ids,
        std::vector<Neighbor> &best_L_nodes, const std::vector<label> &labels_to_filter_by, QueryStats *stats = nullptr);


    void update_degree_stats();

    std::vector<std::mutex>
        _locks;  // Per node lock to be initialized at build time, dont
                 // initialize in constructor to save memory for pure search
  };
}  // namespace grann
