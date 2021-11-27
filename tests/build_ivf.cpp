// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ivf.h>
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
int build_ivf_index(const std::string&   data_path,
                       const unsigned num_clusters,
                       const float training_rate,
                       const std::string& save_path) {
  grann::Parameters paras;
  paras.Set<unsigned>("num_clusters", num_clusters);
  paras.Set<float>("training_rate", training_rate);
  std::vector<_u32> idmap;

  grann::Metric metric = grann::Metric::L2;

  grann::IVFIndex<T> ivf(metric, data_path.c_str(), idmap);
  auto             s = std::chrono::high_resolution_clock::now();
  ivf.build(paras);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  ivf.save(save_path.c_str());

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>] [data_file.bin]  "
                 "[output_index_prefix]  "
              << "[num_clusters] [training_rate] "
              << std::endl;
    exit(-1);
  }




  _u32 ctr = 2;

  const std::string data_path(argv[ctr++]);
  const std::string save_path(argv[ctr++]);
  const unsigned    num_clusters = (unsigned) atoi(argv[ctr++]);
  const float       training_rate = (float) atof(argv[ctr++]);


  if (std::string(argv[1]) == std::string("int8"))
    build_ivf_index<int8_t>(data_path, num_clusters, training_rate, save_path);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_ivf_index<uint8_t>(data_path,  num_clusters, training_rate, save_path);
  else if (std::string(argv[1]) == std::string("float"))
    build_ivf_index<float>(data_path,  num_clusters, training_rate, save_path);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
