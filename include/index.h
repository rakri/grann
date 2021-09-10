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
    ANNIndex(Metric m, const char *filename, std::vector<_u32> &list_of_ids);
    ~ANNIndex();

    virtual void save(const char *filename) = 0;
    virtual void load(const char *filename) = 0;

    virtual void build(Parameters &build_params) = 0;

    // returns # results found (will be <= res_count)
    virtual _u32 search(const T *query, _u32 res_count,
                        Parameters &search_params, _u32 *indices,
                        float *distances, QueryStats *stats = nullptr) = 0;

    /*  Internals of the library */
   protected:
    Metric       _metric = grann::L2;
    Distance<T> *_distance;
    _u32* idmap = nullptr;

    T *    _data;
    size_t _num_points = 0;
    size_t _dim;
    size_t _aligned_dim;  // data dimension is rounded to multiple of 8 for more
                          // efficient alignment and faster floating distance
                          // comparisons. Hence _data is matrix of _num_points *
                          // _aligned_dim size
    bool _has_built = false;
  };
}  // namespace grann
