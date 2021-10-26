#include "hnsw.h"

#define HNSW_FACTOR 1.05

namespace hnsw {
    void select_neighbours(int query, vector<int>& candidate_set, int max_neighbours, void* heuristic) {
        std::vector<int> ret;
        ret.reserve(max_neighbours);

        // ret would be sorted in ascending order

        for(auto element : candidate_set) {
            if(ret.size() < max_neighbours) {
                ret.push_back(element);
                // insert into the right place to maintain the
                // sorted order
            }
            else {
                int index = heurisitic(); // correct place to insert
                ret.pop_back();
                ret.insert(ret.begin() + index, element);                
            }
        }
    }

    int get_random_layer(double normalising_factor) {
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0,1);
        return floor(-log(distribution(generator)) * normalising_factor);
    }

    void build_graph() {
        for(long long i = 0; i < num_points; i++) {
            insert(hnsw_graph, i);
        }
    }

    void insert(vector<layer>& hnsw_graph, int query, ) {
        
    }

    int extract_neighbour(std::set<int>& points, int query, bool neighbour_type) {
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

    vector<grann::Neighbor> search_layer(int query, vector<int> enter_points, int max_return_points, int layer_number) {
        std::vector<grann::Neighbor> ret;

        std::set<int> visited = enter_points;
        std::set<int> candidate_set = enter_points;
        std::set<int> nearest_neighbours = ep;
    
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
}