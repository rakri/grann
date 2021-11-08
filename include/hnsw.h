#pragma once

#include "graph_index.h"
#include "utils.h"
#include "distance.h"

namespace hnsw {
  // typedef layer void;

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
    std::vector<layer> hnsw_graph;

    _u32 num_points;

    _u32 select_neighbors_simple(_u32 query, const std::vector<_u32>& candidates, int max_return, std::vector<_u32>& nearest_neighbors);
    _u32 get_random_layer(double normalising_factor);
    bool build_graph();
    bool insert(std::vector<layer>& hnsw_graph, int query, int num_connections, int max_connections, int ef_value, float normalising_factor);
  };
}  // namespace grann
