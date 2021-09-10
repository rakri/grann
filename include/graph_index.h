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
    GraphIndex(Metric m, const char *filename, std::vector<_u32> &list_of_ids);
    ~GraphIndex();

  //  virtual void save(const char *filename) = 0;
  //  virtual void load(const char *filename) = 0;

//    virtual void build(Parameters &build_params) = 0;

    // returns # results found (will be <= res_count)
//    virtual _u32 search(const T *query, _u32 res_count,
//                        Parameters &search_params, _u32 *indices,
//                        float *distances, QueryStats *stats = nullptr) = 0;

    /*  Internals of the library */
   protected:
    NeighborList _out_nbrs;
    NeighborList _in_nbrs;
    _u32         _max_degree = 0;
    bool         _locks_enabled =
        false;  // will be used at build time, pure search dont need locks

    _u32 process_neighbors_into_candidate_pool(
        const T *&node_coords, std::vector<_u32> &nbr_list,
        std::vector<Neighbor> &best_L_nodes, const _u32 maxListSize,
        _u32 &curListSize, tsl::robin_set<_u32> &inserted_into_pool,
        _u32 &total_comparisons, _u32 &total_hops);

    _u32 greedy_search_to_fixed_point(
        const T *node_coords, const _u32 list_size,
        const std::vector<_u32> &init_ids,
        std::vector<Neighbor> &  expanded_nodes_info,
        tsl::robin_set<_u32> &   expanded_nodes_ids,
        std::vector<Neighbor> &best_L_nodes, QueryStats *stats = nullptr);

    std::vector<std::mutex>
        _locks;  // Per node lock to be initialized at build time, dont
                 // initialize in constructor to save memory for pure search
  };
}  // namespace grann
