- Index class:
    - Member variables:
        1. _metric (like grann::L2)
        2. _distance (distance function)
        3. vec<_u32> _idmap (map from i point to its new id)
        4. T* _data (num_points * aligned_dim)
        5. _u64 _num_points
        6. _u64 _dim, _aligned_dim
        7. bool _has_built

    - Functions: (all are virtual)
        1. save(filename)
        2. load(filename)
        3. build(Parameters)
        4. search()

- Graph class:
    - NeighborList is vec<vec<_u32> >. NeighborList[i] stores neighbors of node $i$.

    - Member variables:
        1. NeighborList _out_nbrs, _in_nbrs
        2. _u32 _max_degree
        3. bool _locks_enabled
        4. vec<mutex> _locks

    - Functions:
        1.  p_u32 process_neighbors_into_candidate_pool(
        const T *&node_coords, std::vector<_u32> &nbr_list,
        std::vector<Neighbor> &best_L_nodes, const _u32 maxListSize,
        _u32 &curListSize, tsl::robin_set<_u32> &inserted_into_pool,
        _u32 &total_comparisons);

        2. void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                         const Parameters &     parameter,
                         std::vector<unsigned> &pruned_list);

        3. _u32 greedy_search_to_fixed_point(
        const T *node_coords, const _u32 list_size,
        const std::vector<_u32> &init_ids,
        std::vector<Neighbor> &  expanded_nodes_info,
        tsl::robin_set<_u32> &   expanded_nodes_ids,
        std::vector<Neighbor> &best_L_nodes, QueryStats *stats = nullptr);

        4. void update_degree_stats();

- Vamana class:
    - Member variables:
        1. _u32 _start_node
    
    - Functions:
        1. _u32 calculate_entry_point();
        2. void get_expanded_nodes(const _u32 node_id, const unsigned l_build,
                              std::vector<_u32>     init_ids,
                              std::vector<Neighbor> &expanded_nodes_info,
                              tsl::robin_set<_u32>  &expanded_nodes_ids);
        3. void inter_insert(_u32 n, std::vector<_u32> &pruned_list,
                        const Parameters &parameters);

- HNSW class:   
    - Member variables:
        1. _u32 _start_node
        2. 

    - Functions:
        1. 