// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.


#include "utils.h"

#include "logger.h"
#include "recall_utils.h"
#include "cached_io.h"
#include "percentile_stats.h"

namespace grann {

  double get_memory_budget(const std::string &mem_budget_str) {
    double mem_ram_budget = atof(mem_budget_str.c_str());
    double final_vamana_ram_limit = mem_ram_budget;
    if (mem_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB >
        THRESHOLD_FOR_CACHING_IN_GB) {  // slack for space used by cached
                                        // nodes
      final_vamana_ram_limit = mem_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
    }
    return final_vamana_ram_limit * 1024 * 1024 * 1024;
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned recall_at) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t    tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        tie_breaker = recall_at - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec,
                 res_vec + recall_at);  // change to recall_at for recall k@k or
                                        // dim_or for k@dim_or
      unsigned cur_recall = 0;
      for (auto &v : gt) {
        if (res.find(v) != res.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_range_search_recall(unsigned num_queries, std::vector<std::vector<_u32>> &groundtruth,
                          std::vector<std::vector<_u32>> &our_results) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();

      gt.insert(groundtruth[i].begin(), groundtruth[i].end());
      res.insert(our_results[i].begin(), our_results[i].end()); 
      unsigned cur_recall = 0;
      for (auto &v : gt) {
        if (res.find(v) != res.end()) {
          cur_recall++;
        }
      }
      if (gt.size() != 0)
      total_recall += ((100.0*cur_recall)/gt.size());
      else
      total_recall += 100;
    }
    return total_recall / (num_queries);
  }


};  // namespace grann
