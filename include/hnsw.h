// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "graph_index.h"
#include "utils.h"
#include "distance.h"

namespace grann {

  template<typename T>
  class HNSW : public GraphIndex<T> {
   public:
    HNSW(Metric m, _u32 level_number, const char *filename,
         std::vector<_u32> &list_of_tags);
    HNSW(Metric m, _u32 level_number);
    ~HNSW();

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
    unsigned _cur_level_number = 0;
    HNSW<T> *_inner_index = nullptr;
  };
}  // namespace grann
