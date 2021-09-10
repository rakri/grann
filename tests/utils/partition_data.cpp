// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <vamana.h>
#include <math_utils.h>
#include "cached_io.h"
#include "partition_and_pq.h"

// DEPRECATED: NEED TO REPROGRAM

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << "Usage:\n"
              << argv[0]
              << "  datatype<int8/uint8/float>  <data_path>"
                 "  <prefix_path>  <sampling_rate>  "
                 "  <num_partitions>  <k_vamana>"
              << std::endl;
    exit(-1);
  }

  const std::string data_path(argv[2]);
  const std::string prefix_path(argv[3]);
  const float       sampling_rate = atof(argv[4]);
  const _u64      num_partitions = (_u64) std::atoi(argv[5]);
  const _u64      max_reps = 15;
  const _u64      k_vamana = (_u64) std::atoi(argv[6]);

  if (std::string(argv[1]) == std::string("float"))
    partition<float>(data_path, sampling_rate, num_partitions, max_reps,
                     prefix_path, k_vamana);
  else if (std::string(argv[1]) == std::string("int8"))
    partition<int8_t>(data_path, sampling_rate, num_partitions, max_reps,
                      prefix_path, k_vamana);
  else if (std::string(argv[1]) == std::string("uint8"))
    partition<uint8_t>(data_path, sampling_rate, num_partitions, max_reps,
                       prefix_path, k_vamana);
  else
    std::cout << "unsupported data format. use float/int8/uint8" << std::endl;
}
