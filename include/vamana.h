// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "graph_index.h"

namespace grann {

  // Inherits from GraphIndex class since Vamana is a graph-based ANNS
  template<typename T>
  class Vamana : public GraphIndex<T> {
    public:
      Vamana(Metric m, const char *filename, std::vector<_u32> &list_of_ids);
      
      ~Vamana();

      // checks if data is consolidated, saves graph, metadata, and associated
      // tags.
      void save(const char *filename);
      
      void load(const char *filename);

      void build(Parameters &parameters);

      // returns #results found (will be <= res_count)
      _u32 search(const T *query, _u32 res_count, Parameters &search_params,
                  _u32 *indices, float *distances, QueryStats *stats = nullptr);

    /*  Internals of the library */
    protected:
      //  _u64   _num_steiner_pts;
      _u32 _start_node;

      _u32 calculate_entry_point();
      void get_expanded_nodes(const _u32 node_id, const unsigned l_build,
                              std::vector<_u32>     init_ids,
                              std::vector<Neighbor> &expanded_nodes_info,
                              tsl::robin_set<_u32>  &expanded_nodes_ids);

      void inter_insert(_u32 n, std::vector<_u32> &pruned_list,
                        const Parameters &parameters);
  };
}  // namespace grann
