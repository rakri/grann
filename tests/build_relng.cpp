// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "relative_ng.h"
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
int build_relng_index(const std::string&   data_path,
                       const grann::Metric& metric,
                       const std::string& save_path,
                       const unsigned     num_threads) {
  grann::Parameters paras;
  paras.Set<unsigned>("num_threads", num_threads);
  std::vector<_u32> idmap;

  grann::RelativeNG<T> relng(metric, data_path.c_str(), idmap);
  auto             s = std::chrono::high_resolution_clock::now();
  relng.build(paras);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  relng.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>] [l2/mips] [data_file.bin]  "
                 "[output_index_file]  "
              << "  [num_threads_to_use]. See README for more information on "
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
  const std::string save_path(argv[ctr++]);
  const unsigned    num_threads = (unsigned) atoi(argv[ctr++]);

  if (std::string(argv[1]) == std::string("int8"))
    build_relng_index<int8_t>(data_path, metric, save_path,
                               num_threads);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_relng_index<uint8_t>(data_path, metric, save_path,
                                num_threads);
  else if (std::string(argv[1]) == std::string("float"))
    build_relng_index<float>(data_path, metric, save_path,
                              num_threads);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
