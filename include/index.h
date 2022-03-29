// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "utils.h"
#include "distance.h"
#include "percentile_stats.h"

namespace grann {

  typedef std::lock_guard<std::mutex>
      LockGuard;  // Use this datastructure to create per vertex locks if we
                  // want to update the graph during index build

  // Neighbor contains infromation of the name of the neighbor and associated
  // distance
  struct SimpleNeighbor {
    _u32  id;
    float distance;

    SimpleNeighbor() = default;
    SimpleNeighbor(unsigned id, float distance) : id(id), distance(distance) {
    }

    inline bool operator<(const SimpleNeighbor &other) const {
      return distance < other.distance;
    }

    inline bool operator>(const SimpleNeighbor &other) const {
      return distance > other.distance;
    }

    inline bool operator==(const SimpleNeighbor &other) const {
      return id == other.id;
    }
  };

  typedef std::vector<SimpleNeighbor> vecNgh;

  // Simple Neighbor with a flag, for remembering whether we already explored
  // out of a vertex or not.
  struct Neighbor {
    _u32  id = 0;
    float distance = std::numeric_limits<float>::max();
    bool  flag = true;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f = true)
        : id{id}, distance{distance}, flag(f) {
    }

    inline bool operator<(const Neighbor &other) const {
      return distance < other.distance;
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
  };

  // Given a  neighbor array of size K starting with pointer addr which is
  // sorted by distance, and a new neighbor nn, insert it in correct place if it
  // can fit. Else dont do anything. If element is already in Pool, returns K+1.
  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K,
                                        Neighbor nn) {
    // find the location to insert
    unsigned left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }
    // check equal ID

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;
      left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    memmove((char *) &addr[right + 1], &addr[right],
            (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
  }

  template<typename T>
  class ANNIndex {
   public:
    // make a base object, initialize distance function and load the data from
    // filename bin file. The list of ids corresponds to the id/tag associated
    // with each vector.
    ANNIndex(Metric m, const char *filename, std::vector<_u32> &list_of_tags);

    //  for loading an index from a file, we dont need data file, and list of
    //  tags
    ANNIndex(Metric m);

    virtual ~ANNIndex();

    virtual void save(const char *filename) = 0;

    virtual void load(const char *filename) = 0;

    virtual void build(Parameters &build_params) = 0;

    // returns # results found (will be <= res_count)
    virtual _u32 search(const T *query, _u32 res_count,
                        Parameters &search_params, _u32 *indices,
                        float *distances, QueryStats *stats = nullptr) = 0;

    /*  Internals of the library */
   protected:
    void save_data_and_tags(const std::string index_file);
    void load_data_and_tags(const std::string index_file);

		typedef std::string label;
    void parse_label_file(std::string map_file);

    _u32 process_candidates_into_best_candidates_pool(
        const T *&node_coords, std::vector<_u32> &nbr_list,
        std::vector<Neighbor> &best_L_nodes, const _u32 maxListSize,
        _u32 &curListSize, tsl::robin_set<_u32> &inserted_into_pool,
        _u32 &total_comparisons);

    unsigned         calculate_medoid_of_data();
		unsigned				 calculate_filtered_medoid();
    Metric           _metric = grann::L2;
    Distance<T> *    _distance;
    Distance<float> *_distance_float;

    _u32 *_tag_map = nullptr;

    T *_data;  // will be a num_points * aligned_dim array stored in row-major
               // form
    _u64 _num_points = 0;  // number of points hosted by index
    _u64 _dim;             // original dimension of data vectors
    _u64 _aligned_dim;  // data dimension is rounded to multiple of 8 for more
                        // efficient alignment and faster floating distance
                        // comparisons. Hence _data is matrix of _num_points *
                        // _aligned_dim size. Remaining ailgned_dim - dim
                        // entries of each vector are padded with zeros.
    bool _has_built = false;

    bool                                  _filtered_index = false;
    std::string                           _search_filter = "";
    std::vector<std::vector<label>> 			_pts_to_labels;
		std::map<label, std::vector<_u32>>		_labels_to_pts;
    tsl::robin_set<label>           			_labels;
    std::string                           _labels_file;
		std::unordered_map<label, _u32> 			_filter_to_medoid_id;
    std::unordered_map<_u32, _u32>        _medoid_counts;
  };
}  // namespace grann
