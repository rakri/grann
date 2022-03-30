// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "utils.h"
#include "distance.h"
#include "percentile_stats.h"
#include "index.h"

namespace grann {

  template<typename T>
  class IVFIndex : public ANNIndex<T> {
   public:
    IVFIndex(Metric m, const char *filename,
						 std::vector<_u32> &list_of_tags);
    IVFIndex(Metric m, std::string labels_fname="");
    ~IVFIndex();

    void save(const char *filename);
    void load(const char *filename);

    void build(Parameters &parameters);

    // returns # results found (will be <= res_count)
    _u32 search(const T *query, _u32 res_count, Parameters &search_params,
                _u32 *indices, float *distances, QueryStats *stats = nullptr,
								std::vector<label> search_filters = std::vector<label>());

    /*  Internals of the library */
   protected:
    _u64                           _num_clusters = 0;
    float *                        _cluster_centers;
    std::string                    _base_file;
    std::vector<std::vector<_u32>> _inverted_index;
  };
}  // namespace grann
