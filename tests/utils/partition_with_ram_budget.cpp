// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vamana.h>
#include <math_utils.h>
#include "cached_io.h"
#include "partition_and_pq.h"

// DEPRECATED: NEED TO REPROGRAM

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << "Usage:\n"
              << argv[0]
              << "  datatype<int8/uint8/float>  <data_path>"
                 "  <prefix_path>  <sampling_rate>  "
                 "  <ram_budget(GB)> <graph_degree>  <k_vamana>"
              << std::endl;
    exit(-1);
  }

  const std::string data_path(argv[2]);
  const std::string prefix_path(argv[3]);
  const float       sampling_rate = atof(argv[4]);
  const double      ram_budget = (double) std::atof(argv[5]);
  const _u64        graph_degree = (_u64) std::atoi(argv[6]);
  const _u64        k_vamana = (_u64) std::atoi(argv[7]);

  if (std::string(argv[1]) == std::string("float"))
    partition_with_ram_budget<float>(data_path, sampling_rate, ram_budget,
                                     graph_degree, prefix_path, k_vamana);
  else if (std::string(argv[1]) == std::string("int8"))
    partition_with_ram_budget<int8_t>(data_path, sampling_rate, ram_budget,
                                      graph_degree, prefix_path, k_vamana);
  else if (std::string(argv[1]) == std::string("uint8"))
    partition_with_ram_budget<uint8_t>(data_path, sampling_rate, ram_budget,
                                       graph_degree, prefix_path, k_vamana);
  else
    std::cout << "unsupported data format. use float/int8/uint8" << std::endl;
}
