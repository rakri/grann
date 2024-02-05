// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vamana.h>
#include <omp.h>
#include <string.h>
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

template<typename T>
int build_vamana_index(const std::string&   data_path, const std::string &labels_file, 
                       const grann::Metric& metric, const unsigned R,
                       const unsigned L, const float alpha, const unsigned avg_degree, 
                       const std::string& save_path,
                       const unsigned     num_threads) {
  grann::Parameters paras;

  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>(
      "C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("avg_degree", avg_degree);
  paras.Set<unsigned>("num_threads", num_threads);

  _u32 C = paras.Get<unsigned>("C");
  _u32 R1 = paras.Get<unsigned>("R");
  _u32 L1 = paras.Get<unsigned>("L");    
  float alpha1 = paras.Get<float>("alpha");
  _u32 pr = paras.Get<uint32_t>("pruning_rule");

  std::cout<<"Parameters set: C=" << C<<", L=" << L1 <<", R=" << R1 <<", alpha=" << alpha1 <<", pruning_rule=" << pr << std::endl;

  std::vector<_u32> idmap;

  grann::Vamana<T> vamana(metric, data_path.c_str(), idmap, labels_file);
  auto             s = std::chrono::high_resolution_clock::now();
  vamana.build(paras);
  vamana.select_most_used_edges(avg_degree, alpha, C);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  vamana.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc < 11 || argc > 12) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>] [l2/mips] [data_file.bin] [filtered_index (0/1)] {labels_file (if filtered_index==1)}  "
                 "[output_vamana_file]  "
              << "[R]  [L]  [alpha]"
              << "  [num_threads_to_use] [avg_degree]. See README for more information on "
                 "parameters."
              << std::endl;
    exit(-1);
  }

  _u32 ctr = 2;

  grann::Metric metric;
  if (std::string(argv[ctr]) == std::string("mips"))
    metric = grann::Metric::INNER_PRODUCT;
  else if (std::string(argv[ctr]) == std::string("l2"))
    metric = grann::Metric::L2;
  else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product support."
              << std::endl;
    return -1;
  }
  ctr++;

  const std::string data_path(argv[ctr++]);

  bool filtered_index = (bool) std::atoi(argv[ctr++]);
  std::string labels_file = "";
  if (filtered_index) {
	labels_file = std::string(argv[ctr++]);
  }


  const std::string save_path(argv[ctr++]);
  const unsigned    R = (unsigned) atoi(argv[ctr++]);
  const unsigned    L = (unsigned) atoi(argv[ctr++]);
  const float       alpha = (float) atof(argv[ctr++]);
  const unsigned    num_threads = (unsigned) atoi(argv[ctr++]);
  const unsigned    avg_degree = (unsigned) atoi(argv[ctr++]);


  if (std::string(argv[1]) == std::string("int8"))
    build_vamana_index<int8_t>(data_path, labels_file, metric, R, L, alpha, avg_degree, save_path, 
                               num_threads);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_vamana_index<uint8_t>(data_path, labels_file,  metric, R, L, alpha, avg_degree, save_path,
                                num_threads);
  else if (std::string(argv[1]) == std::string("float"))
    build_vamana_index<float>(data_path,  labels_file, metric, R, L, alpha, avg_degree, save_path,
                              num_threads);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
