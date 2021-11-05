#pragma once

#include "distance.h"
#include "graph_index.h"
#include "index.h"
#include "utils.h"

namespace grann {

  template<typename T>
  class Hnsw : public ANNIndex<T> {
    public:
    //TODO: Add in documentation for these public functions
    Hnsw(Metric m, const char *filename, std::vector<_u32> &list_of_ids); // maybe dont need list_of_ids

    ~Hnsw();

    void save(const char *filename);

    void load(const char * filename);

    void build(Parameters &parameters);

    //TODO: parameters need work
    _u32 search(const T *query, _u32 res_count,
                          Parameters &search_params, _u32 *indices,
                          float *distances, QueryStats *stats = nullptr);

   protected:
    // We encode the entire index within a vector of Layer constructs.
    std::vector<Layer<T>> hnsw_graph;

    // The total number of points in the index (including duplicates).
    long long num_points;

    /*
     * High level function to build the graph.
     *
     * */
    //TODO: figure out parameters (probably dependent on the public build function)
    void build_graph();

    //NOTE: what is this for again?
    void sample_points();
  }
}  // namespace grann
