// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "graph_index.h"
#include "utils.h"
#include "distance.h"

namespace grann {

  template<typename T>
  class RelativeNG : public GraphIndex<T> {
   public:
    RelativeNG(Metric m, const char *filename, std::vector<_u32> &list_of_tags);
    //    ~RelativeNG();

    // checks if data is consolidated, saves graph, metadata and associated
    // tags.
    void save(const char *filename);
    void load(const char *filename);

    void build(Parameters &parameters);

    // returns # results found (will be <= res_count)
    _u32 search(const T *query, _u32 res_count, Parameters &search_params,
                _u32 *indices, float *distances, QueryStats *stats = nullptr);

    /*  Internals of the library */
   protected:
    //  _u64   _num_steiner_pts;
    unsigned _start_node;

    unsigned calculate_entry_point();
  };
}  // namespace grann
