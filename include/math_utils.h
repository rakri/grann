// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <mkl.h>

#include "utils.h"

namespace math_utils {

  float calc_distance(float* vec_1, float* vec_2, _u64 dim);

  // compute l2-squared norms of data stored in row major num_points * dim,
  // needs
  // to be pre-allocated
  void compute_vecs_l2sq(float* vecs_l2sq, float* data, const _u64 num_points,
                         const _u64 dim);

  void rotate_data(float* data, _u64 num_points, _u64 dim,
                            float* rot_mat, float*& new_mat,
                            bool transpose_rot = false);

  // calculate closest center to data of num_points * dim (row major)
  // centers is num_centers * dim (row major)
  // data_l2sq has pre-computed squared norms of data
  // centers_l2sq has pre-computed squared norms of centers
  // pre-allocated center_ids will contain id of k nearest centers
  // pre-allocated dist_matrix shound be num_points * num_centers and contain
  // squared distances

  // Ideally used only by compute_closest_centers
  void compute_closest_centers_in_block(
      const float* const data, const _u64 num_points, const _u64 dim,
      const float* const centers, const _u64 num_centers,
      const float* const docs_l2sq, const float* const centers_l2sq,
      uint32_t* center_ids, float* const dist_matrix, _u64 k = 1);

  // Given data in num_points * new_dim row major
  // Centers stored in centers as k * new_dim row major
  // Calculate the closest center for each point and store it in vector
  // closest_centers_ivf (which needs to be allocated outside)
  // Additionally, if inverted index is not null (and pre-allocated), it will
  // return inverted index for each center Additionally, if pts_norms_squared
  // is not null, then it will assume that point norms are pre-computed and use
  // those
  // values

  void compute_closest_centers(float* data, _u64 num_points, _u64 dim,
                               float* centers, _u64 num_centers, _u64 k,
                               uint32_t*            closest_centers_ivf,
                               std::vector<_u64>* inverted_index = nullptr,
                               float*               pts_norms_squared = nullptr);

  // if to_subtract is 1, will subtract nearest center from each row. Else will
  // add. Output will be in data_load iself.
  // Nearest centers need to be provided in closst_centers.

  void process_residuals(float* data_load, _u64 num_points, _u64 dim,
                         float* centers, _u64 num_centers,
                         uint32_t* closest_centers, bool to_subtract);


  // run Lloyds one iteration
  // Given data in row major num_points * dim, and centers in row major
  // num_centers * dim
  // And squared lengths of data points, output the closest center to each data
  // point, update centers, and also return inverted vamana.
  // If closest_centers == nullptr, will allocate memory and return.
  // Similarly, if closest_docs == nullptr, will allocate memory and return.

  float lloyds_iter(float* data, _u64 num_points, _u64 dim, float* centers,
                    _u64 num_centers, float* docs_l2sq,
                    std::vector<_u64>* closest_docs,
                    uint32_t*&           closest_center);

  // Run Lloyds until max_reps or stopping criterion
  // If you pass nullptr for closest_docs and closest_center, it will NOT return
  // the results, else it will assume appriate allocation as closest_docs = new
  // vector<_u64> [num_centers], and closest_center = new _u64[num_points]
  // Final centers are output in centers as row major num_centers * dim
  //
  float run_lloyds(float* data, _u64 num_points, _u64 dim, float* centers,
                   const _u64 num_centers, const _u64 max_reps,
                   std::vector<_u64>* closest_docs, uint32_t* closest_center);

  // assumes already memory allocated for center_data as new
  // float[num_centers*dim] and select randomly num_centers points as centers
  void random_centers(float* data, _u64 num_points, _u64 dim,
                        float* centers, _u64 num_centers);

  void kmeans_plus_plus_centers(float* data, _u64 num_points, _u64 dim,
                                 float* centers, _u64 num_centers);
}; // namespace math_utils
