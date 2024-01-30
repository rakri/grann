// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <omp.h>
#include <set>
#include <string.h>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "recall_utils.h"
#include "ivf.h"
#include "utils.h"

template<typename T>
int search_index(int argc, char** argv) {
  T*                query = nullptr;
  unsigned*         gt_ids = nullptr;
  float*            gt_dists = nullptr;
  _u64              query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  grann::Metric metric = grann::Metric::L2;

  _u32 ctr = 2;

  std::string index_file(argv[ctr++]);
  _u64        num_threads = std::atoi(argv[ctr++]);
  std::string query_bin(argv[ctr++]);
  std::string truthset_bin(argv[ctr++]);
  _u64        recall_at = std::atoi(argv[ctr++]);
  std::string result_output_prefix(argv[ctr++]);
  bool filtered_search = (bool) std::atoi(argv[ctr++]);
  std::vector<grann::label> search_filters;
  if (filtered_search) {
      std::istringstream       iss(argv[ctr++]);
      std::string token;
      while (getline(iss, token, ',')) {
        token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
        search_filters.push_back(token);
      }

//	search_filters = {argv[ctr++]};
  }

  bool calc_recall_flag = false;

  for (; ctr < (_u32) argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= 1)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid probe width found -- must be at 1." << std::endl;
    return -1;
  }

  grann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                             query_aligned_dim);

  if (file_exists(truthset_bin)) {
    grann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  grann::IVFIndex<T> ivf_index(metric);
  ivf_index.load(index_file.c_str());  // to load Index
  std::cout << "IVF Index loaded" << std::endl;
  grann::Parameters search_params;

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(22)
            << "Mean Latency (mus)" << std::setw(15) << "99.9\% Latency"
            << std::setw(12) << recall_string << std::setw(16) << "Mean Cmps."
            << std::setw(12) << "Mean Hops" << std::setw(16) << "99.9\% Cmps."
            << std::setw(12) << "99.9\% Hops" << std::endl;
  std::cout << "==============================================================="
               "=========================================================="
            << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  std::vector<double> latency_stats(query_num, 0);

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];
    search_params.Set<_u32>("probe_width", L);
    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    std::vector<grann::QueryStats> stats(query_num);
    auto s = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
//    query_num = 1;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      ivf_index.search(query + i * query_aligned_dim, recall_at, search_params,
                       query_result_ids[test_id].data() + i * recall_at,
                       query_result_dists[test_id].data() + i * recall_at,
                       (stats.data() + i), search_filters);
      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000000;
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    float qps = (query_num / diff.count());

    float recall = 0;
    if (calc_recall_flag)
      recall = grann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                       query_result_ids[test_id].data(),
                                       recall_at, recall_at);

    std::sort(latency_stats.begin(), latency_stats.end());
    double mean_latency = 0;
    for (uint64_t q = 0; q < query_num; q++) {
      mean_latency += latency_stats[q];
    }
    mean_latency /= query_num;

    float mean_cmps = grann::get_mean_stats(
        stats.data(), query_num,
        [](const grann::QueryStats& stats) { return stats.n_cmps; });

    float mean_hops = grann::get_mean_stats(
        stats.data(), query_num,
        [](const grann::QueryStats& stats) { return stats.n_hops; });

    float cmps_999 = grann::get_percentile_stats(
        stats.data(), query_num, 0.999,
        [](const grann::QueryStats& stats) { return stats.n_cmps; });

    float hops_999 = grann::get_percentile_stats(
        stats.data(), query_num, 0.999,
        [](const grann::QueryStats& stats) { return stats.n_hops; });

    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(22)
              << (float) mean_latency << std::setw(15)
              << (float) latency_stats[(_u64)(0.999 * query_num)]
              << std::setw(12) << recall << std::setw(16) << mean_cmps
              << std::setw(12) << mean_hops << std::setw(16) << cmps_999
              << std::setw(12) << hops_999 << std::endl;
  }

  std::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    std::string cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    grann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                          query_num, recall_at);
    test_id++;
  }

  grann::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 10) {
    std::cout
        << "Usage: " << argv[0]
        << "  [data_type<float/int8/uint8>]  "
           "[index_prefix]  [num_threads] "
           "[query_file.bin]  [truthset.bin (use \"null\" for none)] "
           " [K] [result_output_prefix] [filtered_search (0/1)] {filter_label (if filtered_search==1)} "
           " [P1]  [P2] etc. See README for more information on parameters. "
        << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("int8"))
    search_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_index<uint8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("float"))
    search_index<float>(argc, argv);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
