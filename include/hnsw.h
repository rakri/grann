#pragma once

#include "graph_index.h"
#include "utils.h"
#include "distance.h"

namespace grann {
  typedef layer void;

  template<typename T>
  class Hnsw : public GraphIndex<T> {
   public:
    Hnsw(Metric m, const char *filename, std::vector<_u32> &list_of_ids); // maybe dont need list_of_ids

    ~Hnsw();

    void save(const char *filename);

    void load(const char * filename);

    void build(Parameters &parameters);
    
    _u32 search(const T *query, _u32 res_count,
                          Parameters &search_params, _u32 *indices,
                          float *distances, QueryStats *stats = nullptr);

   protected:
    // vector of layers
    vector<layer> hnsw_graph;

    long long num_points;

    void build_graph();
    void search_layer();  // can make it iterative too
    void sample_points();
  }
}  // namespace grann
