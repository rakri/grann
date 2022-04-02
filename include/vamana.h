// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <unordered_set>
#include "graph_index.h"
#include "utils.h"
#include "distance.h"

namespace grann {

  template<typename T>
  class Vamana : public GraphIndex<T> {
   public:
    Vamana(Metric m, const char *filename, std::vector<_u32> &list_of_tags, std::string labels_fname="");
    Vamana(Metric m);
    //    ~Vamana();

    // checks if data is consolidated, saves graph, metadata and associated
    // tags.
    void save(const char *filename);
    void load(const char *filename);

    void build(const Parameters &parameters);


    // returns # results found (will be <= res_count)
    _u32 search(const T *query, _u32 res_count, const Parameters &search_params,
                _u32 *indices, float *distances, QueryStats *stats = nullptr,
								std::vector<label> search_filters = std::vector<label>());

    /*  Internals of the library */
   protected:
    //  _u64   _num_steiner_pts;
    unsigned _start_node;
    // for filtered indices
    std::unordered_map<label, _u32>    _filter_to_medoid_id;
    std::unordered_map<_u32, _u32>     _medoid_counts;    

    void calculate_label_specific_medoids();

    void get_expanded_nodes(const _u64 node_id, const unsigned l_build,
                            std::vector<unsigned>     init_ids,
                            std::vector<Neighbor> &   expanded_nodes_info,
                            tsl::robin_set<unsigned> &expanded_nodes_ids, const std::vector<label> &labels_to_accept = std::vector<label>());


  };
}  // namespace grann
