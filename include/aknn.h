// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "graph_index.h"
#include "vamana.h"
#include "utils.h"
#include "distance.h"

namespace grann {

  template<typename T>
  class ApproxKNN : public GraphIndex<T> {
   public:
    ApproxKNN(Metric m, const char *filename, std::vector<_u32> &list_of_tags, std::string labels_fname="");
    ApproxKNN(Metric m);
    //    ~ApproxKNN();

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


  Vamana<T>* _vamana_for_build;
  std::string _data_file_path;

  };
}  // namespace grann
