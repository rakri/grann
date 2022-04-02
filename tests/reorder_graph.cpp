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
#include "vamana.h"
#include "utils.h"

template<typename T>
int reorder_graph(int argc, char** argv) {

  grann::Metric metric = grann::Metric::L2;
  _u32          ctr = 2;
  std::string vamana_file(argv[ctr++]);
  _u64        num_threads = std::atoi(argv[ctr++]);
  _u32 omega = std::atoi(argv[ctr++]);
  std::string output_prefix(argv[ctr++]);
 
  grann::Vamana<T> vamana(metric);
  vamana.load(vamana_file.c_str());  // to load Vamana Index
  std::cout << "Vamana loaded" << std::endl;

  vamana.reorder(output_prefix, omega, num_threads);
  
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout
        << "Usage: " << argv[0] <<
              " [data_type (int8/uint8/float)] [input_graph_index_path]  [num_threads] "
           "[omega]  [output_prefix] "
        << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("int8"))
    reorder_graph<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    reorder_graph<uint8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("float"))
    reorder_graph<float>(argc, argv);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
