// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "utils.h"
#include "distance.h"
#include "percentile_stats.h"

namespace grann {

  typedef std::lock_guard<std::mutex>
      LockGuard;  // Use this data structure to create per vertex locks if we
                  // want to update the graph during index build

  // Neighbor contains information of the name of the neighbor and associated
  // distance
  class SimpleNeighbor {
    protected:
      _u32  id;
      float distance;

    public:
      SimpleNeighbor() = default;

      SimpleNeighbor(_u32 id, float distance) : id(id), distance(distance) {
      }

      inline bool operator<(const SimpleNeighbor &other) const {
        return this.distance < other.distance;
      }

      inline bool operator>(const SimpleNeighbor &other) const {
        return this.distance > other.distance;
      }

      inline bool operator==(const SimpleNeighbor &other) const {
        return this.id == other.id;
      }
  };

  typedef std::vector<SimpleNeighbor> vecNgh;

  // Simple Neighbor with a flag, for remembering whether we have already
  // explored out of a vertex or not.
  class Neighbor : public SimpleNeighbor {
    private:
      bool  flag = true;

    public:
      // default constructor
      Neighbour() 
          : id(0), distance(std::numeric_limits<float>::max()), flag(true) {
      }

      Neighbor(_u32 id, float distance, bool flag = true)
          : id{id}, distance{distance}, flag(flag) {
      }
  };

  // Given an array of neighbours of size K sorted by distance (ascending), and
  // a new neighbor nn, insert it in the correct place if it can fit. 
  // Else do not do anything. 
  // If element is already in the Pool, return K+1.
  static inline _u32 InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    _u32 left = 0, right = K - 1;
    
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    else if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }

    while (right > 1 && left < right - 1) {
      _u32 mid = left + (right - left) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id) {
        // already exists in the Pool
        return K + 1;
      }
      left--;
    }

    if (addr[left].id == nn.id || addr[right].id == nn.id) return K + 1;
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
      ANNIndex(Metric m, const char *filename, const std::vector<_u32> &list_of_ids);

      ~ANNIndex();

      // function_name() = 0 creates a pure virtual function.
      // Hence, this class is an abstract class and cannot be instantiated.
      virtual void save(const char *filename) = 0;
      virtual void load(const char *filename) = 0;
      virtual void build(Parameters &build_params) = 0;

      // returns #results found (should be <= res_count)
      virtual _u32 search(const T *query, _u32 res_count,
                          Parameters &search_params, _u32 *indices,
                          float *distances, QueryStats *stats = nullptr) = 0;

    /*  Internals of the library */
    protected:
      Metric       _metric = grann::L2;
      Distance<T> *_distance;
      _u32 *       idmap = nullptr;

      T *  _data; // num_points * aligned_dim array stored in row-major form
      _u64 _num_points = 0; // number of points hosted by index
      _u64 _dim; // original dimension of data vectors
      _u64 _aligned_dim;  // data dimension is rounded to multiple of 8 for more
                          // efficient alignment and faster floating distance
                          // comparisons. Hence _data is matrix of _num_points *
                          // _aligned_dim size. Remaining ailgned_dim - dim entries
                          // of each vector are padded with zeros.
      bool _has_built = false;
  };
}  // namespace grann
