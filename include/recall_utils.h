// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else
#include <unistd.h>
typedef int FileHandle;
#endif

#include "cached_io.h"
#include "essentials.h"
#include "utils.h"

#include "gperftools/malloc_extension.h"

namespace grann {
  const _u64   TRAINING_SET_SIZE = 100000;
  const double   SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
  const double   THRESHOLD_FOR_CACHING_IN_GB = 1.0;
  const uint32_t NUM_NODES_TO_CACHE = 250000;
  const uint32_t WARMUP_L = 20;
  const uint32_t NUM_KMEANS_REPS = 12;

  template<typename T>
  class DiskANN;

  inline void load_truthset(const std::string& bin_file, uint32_t*& ids,
                            float*& dists, _u64& npts, _u64& dim) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    grann::cout << "Reading truthset file " << bin_file.c_str() << " ..."
                << std::endl;
    _u64 actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    grann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
                << std::endl;

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    _u64 expected_file_size_with_dists =
        2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    _u64 expected_file_size_just_ids =
        npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
      truthset_type = 2;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size
             << ", expected: " << expected_file_size_with_dists << " or "
             << expected_file_size_just_ids;
      grann::cout << stream.str();
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
    }

    ids = new uint32_t[npts * dim];
    reader.read((char*) ids, npts * dim * sizeof(uint32_t));

    if (truthset_type == 1) {
      dists = new float[npts * dim];
      reader.read((char*) dists, npts * dim * sizeof(float));
    }
  }

  inline void prune_truthset_for_range(
      const std::string& bin_file, float range,
      std::vector<std::vector<_u32>>& groundtruth, _u64& npts) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    grann::cout << "Reading truthset file " << bin_file.c_str() << " ..."
                << std::endl;
    _u64 actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    _u64   dim = (unsigned) dim_i32;
    _u32*  ids;
    float* dists;

    grann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..."
                << std::endl;

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    _u64 expected_file_size_with_dists =
        2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
      truthset_type = 1;

    if (truthset_type == -1) {
      std::stringstream stream;
      stream << "Error. File size mismatch. File should have bin format, with "
                "npts followed by ngt followed by npts*ngt ids and optionally "
                "followed by npts*ngt distance values; actual size: "
             << actual_file_size
             << ", expected: " << expected_file_size_with_dists;
      grann::cout << stream.str();
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
    }

    ids = new uint32_t[npts * dim];
    reader.read((char*) ids, npts * dim * sizeof(uint32_t));

    if (truthset_type == 1) {
      dists = new float[npts * dim];
      reader.read((char*) dists, npts * dim * sizeof(float));
    }
    float min_dist = std::numeric_limits<float>::max();
    float max_dist = 0;
    groundtruth.resize(npts);
    for (_u32 i = 0; i < npts; i++) {
      groundtruth[i].clear();
      for (_u32 j = 0; j < dim; j++) {
        if (dists[i * dim + j] <= range) {
          groundtruth[i].emplace_back(ids[i * dim + j]);
        }
        min_dist =
            min_dist > dists[i * dim + j] ? dists[i * dim + j] : min_dist;
        max_dist =
            max_dist < dists[i * dim + j] ? dists[i * dim + j] : max_dist;
      }
      // std::cout<<groundtruth[i].size() << " " ;
    }
    std::cout << "Min dist: " << min_dist << ", Max dist: " << max_dist
              << std::endl;
    delete[] ids;
    delete[] dists;
  }

  inline void load_range_truthset(const std::string&              bin_file,
                                  std::vector<std::vector<_u32>>& groundtruth,
                                  _u64&                           gt_num) {
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    grann::cout << "Reading truthset file " << bin_file.c_str() << " ..."
                << std::endl;
    _u64 actual_file_size = reader.get_file_size();

    int npts_u32, total_u32;
    reader.read((char*) &npts_u32, sizeof(int));
    reader.read((char*) &total_u32, sizeof(int));

    gt_num = (_u64) npts_u32;
    _u64 total_res = (_u64) total_u32;

    grann::cout << "Metadata: #pts = " << gt_num
                << ", #total_results = " << total_res << "..." << std::endl;

    _u64 expected_file_size =
        2 * sizeof(_u32) + gt_num * sizeof(_u32) + total_res * sizeof(_u32);

    if (actual_file_size != expected_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch in range truthset. actual size: "
             << actual_file_size << ", expected: " << expected_file_size;
      grann::cout << stream.str();
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
    }
    groundtruth.clear();
    groundtruth.resize(gt_num);
    std::vector<_u32> gt_count(gt_num);

    reader.read((char*) gt_count.data(), sizeof(_u32) * gt_num);

    std::vector<_u32> gt_stats(gt_count);
    std::sort(gt_stats.begin(), gt_stats.end());

    std::cout << "GT count percentiles:" << std::endl;
    for (_u32 p = 0; p < 100; p += 5)
      std::cout << "percentile " << p << ": "
                << gt_stats[std::floor((p / 100.0) * gt_num)] << std::endl;
    std::cout << "percentile 100"
              << ": " << gt_stats[gt_num - 1] << std::endl;

    for (_u32 i = 0; i < gt_num; i++) {
      groundtruth[i].clear();
      groundtruth[i].resize(gt_count[i]);
      if (gt_count[i] != 0)
        reader.read((char*) groundtruth[i].data(), sizeof(_u32) * gt_count[i]);
    }
  }

   double calculate_recall(unsigned  num_queries,
                                          unsigned* gold_std, float* gs_dist,
                                          unsigned  dim_gs,
                                          unsigned* our_results,
                                          unsigned dim_or, unsigned recall_at);

   double calculate_range_search_recall(
      unsigned num_queries, std::vector<std::vector<_u32>>& groundtruth,
      std::vector<std::vector<_u32>>& our_results);

}  // namespace grann
