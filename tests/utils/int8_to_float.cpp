// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " input_int8_bin output_float_bin" << std::endl;
    exit(-1);
  }

  int8_t* input;
  _u64  npts, nd;
  grann::load_bin<int8_t>(argv[1], input, npts, nd);
  float* output = new float[npts * nd];
  grann::convert_types<int8_t, float>(input, output, npts, nd);
  grann::save_bin<float>(argv[2], output, npts, nd);
  delete[] output;
  delete[] input;
}
