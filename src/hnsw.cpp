// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define HNSW_FACTOR 1.05

#include "hnsw.h"

namespace grann {
    // constructor
    template<typename T>
    HNSW<T>::HNSW(Metric m, const char *filename,
                            std::vector<_u32> &list_of_ids) : ANNIndex<T>(m, filename, list_of_ids) {
        grann::cout << "Initialized HNSW object with " << this->_num_points << " points, dim = " << this->_dim << "." << std::endl;
    }

    template<typename T>
    void HNSW<T>::save(const char *filename) {
        
    }

    template<typename T>
    void HNSW<T>::load(const char *filename) {
        
    }

    template<typename T>
    void HNSW<T>::build(Parameters &parameters) {
        
    }

    template<typename T>
    _u32 HNSW<T>::search(const T *query, _u32 res_count, Parameters &search_params, _u32 *indices, float *distances, QueryStats *stats = nullptr) {
        
    }

    // get a random layer with exponentially decreasing distribution
    _u32 get_random_layer(float normalising_factor) {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0,1);
        return floor(-log(distribution(generator)) * normalising_factor);
    }

    // insert node i into the HNSW graph
    // returns final status of insertion
    template <typename T>
    bool HNSW<T>::insert_node(_u32 query, _u32 num_connections, _u32 ef_construction, float normalising_factor) {
        std::vector<_u32> nearest_neighbours;
        _u32 ep = this->entry_point;
        int top_level = this->num_layers - 1;
        int new_element_level = this->get_random_layer(normalising_factor);
        std::vector<_u32> entry_points;

        for(int l = top_level; l > new_element_level; l--) {
            entry_points = { ep };
            nearest_neighbours = search_layer(query, entry_points, 1, l);
            entry_point = extract_neighbour(nearest_neighbours, query, 0);
        }

        for(int l = min(top_level, new_element_level); l >= 0; l--) {
            nearest_neighbors = search_layer(query, entry_points, ef_value, l);
            std::vector<_u32> neighbours = select_neighbors_simple(queery, nearest_neighbours, num_connections, l);

            for(auto e : neighbours) {
                std::vector<_u32> neighborhood_e = get_neighborhood(e, l);
                if(neighborhood_e.size() > max_connections) {
                    neighborhood_e = select_neighbors_simple(e, neighborhood_e, max_connections, l);
                    set_neighborhood(e, l, neighborhood_e);
                }
            }

            entry_points = nearest_neighbors;
        }

        if(new_element_level > top_level) {
            this->entry_point = query;
        }

        return true;
    }

    // returns the size of set of nearest neighbours
    template<typename T>
    _u32 HNSW<T>::select_neighbors_simple(_u32 query, const std::vector<_u32>& candidates, int max_return, std::vector<_u32>& nearest_neighbors) {
        int num_elements = 0;
        nearest_neighbors.resize(0);

        std::vector<float> neighbor_distance;

        for(_u32 candidate : candidates) {
            float dist = distance(query, candidate);
            auto it = std::lower_bound(neighbor_distance.begin(), neighbor_distance.end(), dist);

            neighbor_distance.insert(it, dist);
            auto diff = std::distance(neighbor_distance.begin(), it);
            nearest_neighbors.insert(nearest_neighbors.begin() + diff, candidate);

            if(num_elements == max_return) {
                neighbor_distance.pop_back();
                nearest_neighbors.pop_back();
            }
            else num_elements++;
        }

        return num_elements;
    }

    // returns status of build
    template <typename T>
    bool HNSW<T>::build_graph() {
        for(long long i = 0; i < this->num_points; i++) {
            this->insert(hnsw_graph, i);
        }

        return true;
    }

    int extract_neighbour(std::vector<int>& points, int query, bool neighbour_type) {
        int ret = points[0];
        if(neighbour_type == 0) {
            // nearest neighbour
            for(auto p : points) {

            }
        }
        else {
            // furthest neighbour
            for(auto p : points) {

            }
        }

        return ret;
    }

    std::vector<grann::Neighbor> search_layer(int query, std::vector<int> enter_points, int max_return_points, int layer_number) {
        std::vector<grann::Neighbor> ret;

        // convert enter_points into a set

        std::set<int> visited = enter_points;
        std::set<int> candidate_set = enter_points;
        std::set<int> nearest_neighbours = enter_points;
    
        while(candidate_set.size() > 0) {
            int c = extract_neighbour(candidate_set, query, 0);
            int f = extract_neighbour(nearest_neighbours, query, 1);

            if(dist(c, query) > dist(f, query)) {
                break;
            }

            for(auto e : hnsw[layer_number].get_neighbours(c)) {
                if(!visited.count(e)) {
                    visited.insert(e);
                    f = extract_neighbour(nearest_neighbour, query, 1);

                    if(dist(e, query) < distance(f, query) || nearest_neighbours.size() < max_return_points) {
                        candidate_set.insert(e);
                        nearest_neighbours.insert(e);

                        if(nearest_neighbours.size() > max_return_points) {
                            nearest_neighbours.erase(f);
                        }
                    }
                }
            }
        }

        return std::vector<int>(nearest_neighbour.begin(), nearest_neighbour.end());
    }

    // EXPORTS
    template class HNSW<float>;
    template class HNSW<int8_t>;
    template class HNSW<uint8_t>;
} // namespace grann