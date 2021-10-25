#include "hnsw.h"

#define HNSW_FACTOR 1.05

namespace hnsw {
    void select_neighbours(int query, vector<int>& candidate_set, int max_neighbours, void* heuristic) {
        vector<int> ret;
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

    void build_graph() {
        for(long long i = 0; i < num_points; i++) {
            insert(hnsw_graph, i);
        }
    }

    void insert(vector<layer>& hnsw_graph, int query, ) {
        
    }
}