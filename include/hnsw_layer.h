#pragma once

#include "graph_index.h"

namespace grann {
    /* Represents a layer in the HNSW graph. */

    template<typename T>
    class Layer : public GraphIndex<T>{
        public:
            /*
             * Constructor for the Layer object.
             *
             * @param: m Metric used for distances between points
             * @param: layer_num Which layer within the HNSW index
             * @param: max_connections Maximum number of edges in the Layer graph
             * @param: max_degree Maximum degree per node in the Layer graph
             * */
            Layer(Metric m, const _u32 layer_num,
                  const _u32 max_connections, const _u32 max_degree);


            /* Destructor for layer object */
            ~Layer();


            /*
             * Inserts node into Layer object.
             *
             * This is only used to build the Layer graph.
             *
             * @param: insert_point The point to be inserted into the graph
             * */
            bool insert_node(const T *insert_point);


            /*
             * Searches Layer for closest point to given query.
             *
             * This is used for both searching and building the Layer graph.
             *
             * @param: query_pt The point whose neighbors we want
             * @param: entry_pt The point we start the search at
             * @param: max_return_size The number of neighbors to return
             * */
            _u32 search_layer(const T *query_pt, const T *entry_pt,
                              _u32 max_return_size);

        protected:
            // Maximum number of total edges & out-edges per node, respectively.
            unsigned max_connections, max_degree;


            /*
             * Chooses the neighbors to draw edges to from the current node.
             *
             * Used in building the Layer graph.
             *
             * @param: query_pt The current node to find neighbors for
             * @param: candidate_set The set of candidates for neighbors
             * @param: max_neighbors The maximum number of neighbors to be returned
             *
             * */
            void select_neighbors(const T *query_pt, std::vector<_u32>& candidate_set,
                                  _u32 max_neighbors);

  };
}  // namespace grann
