// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "graph_index.h"

namespace grann {
  template<typename T>
  class HNSW : public ANNIndex<T> {
   public:
    HNSW(Metric m, const char *filename, std::vector<_u32> &list_of_ids);

    ~Hnsw();

    void save(const char *filename);

    void load(const char *filename);

    void build(Parameters &parameters);

    _u32 search(const T *query, _u32 res_count, Parameters &search_params,
                _u32 *indices, float *distances, QueryStats *stats = nullptr);

   protected:
    // field variables
    _u32 _num_layers;                               // number of layers
    std::vector<GraphIndex*> layers;                // HNSW graph
    _u32 _max_degree;                               // maximum degree at each layer
    _u32 _max_degree_base;                          // maximum degree at base layer
    _u32 _start_node;                               // entry point into the HNSW graph

    // member functions 
    bool insert_node(_u32 query, _u32 num_connections, _u32 ef_construction, float normalising_factor);

    std::vector<_u32> search_layer(_u32 query, std::vector<_u32> enter_points, _u32 ef, _u32 layer_number);
    
    _u32 select_neighbors_simple(_u32 query, const std::vector<_u32>& candidate_set, _u32 max_return_size, std::vector<_u32> &nearest_neighbors);

    _u32 select_neighbors_heuristic(_u32 query, const std::vector<_u32>& candidate_set, _u32 max_return_size, std::vector<_u32> &nearest_neighbors, bool extend_candidates, bool keep_pruned);
  };
}  // namespace grann
