// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "essentials.h"
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#include <unistd.h>
typedef int FileHandle;

#include "logger.h"
#include "cached_io.h"

namespace grann {

  inline void alloc_aligned(void** ptr, _u64 size, _u64 align) {
    *ptr = nullptr;
    assert(IS_ALIGNED(size, align));
#ifndef _WINDOWS
    *ptr = ::aligned_alloc(align, size);
#else
    *ptr = ::_aligned_malloc(size, align);  // note the swapped arguments!
#endif
    assert(*ptr != nullptr);
  }

  inline void aligned_free(void* ptr) {
    // Gopal. Must have a check here if the pointer was actually allocated by
    // _alloc_aligned
    if (ptr == nullptr) {
      return;
    }
#ifndef _WINDOWS
    free(ptr);
#else
    ::_aligned_free(ptr);
#endif
  }

  // get_bin_metadata functions START
  inline void get_bin_metadata_impl(std::basic_istream<char>& reader,
                                    _u64& nrows, _u64& ncols) {
    int nrows_32, ncols_32;
    reader.read((char*) &nrows_32, sizeof(int));
    reader.read((char*) &ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
  }

  inline void get_bin_metadata(const std::string& bin_file, _u64& nrows,
                               _u64& ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    get_bin_metadata_impl(reader, nrows, ncols);
  }
  // get_bin_metadata functions END

  template<typename T>
  inline std::string getValues(T* data, _u64 num) {
    std::stringstream stream;
    stream << "[";
    for (_u64 i = 0; i < num; i++) {
      stream << std::to_string(data[i]) << ",";
    }
    stream << "]" << std::endl;

    return stream.str();
  }

  // load_bin functions START
  template<typename T>
  inline void load_bin_impl(std::basic_istream<char>& reader,
                            _u64 actual_file_size, T*& data, _u64& npts,
                            _u64& dim) {
    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    grann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "."
                << std::endl;

    _u64 expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch. Actual size is " << actual_file_size
             << " while expected size is  " << expected_actual_file_size
             << " npts = " << npts << " dim = " << dim
             << " size of <T>= " << sizeof(T) << std::endl;
      grann::cout << stream.str();
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
    }

    data = new T[npts * dim];
    reader.read((char*) data, npts * dim * sizeof(T));
  }

  template<typename T>
  inline void load_bin(const std::string& bin_file, T*& data, _u64& npts,
                       _u64& dim) {
    grann::cout << "Reading bin file " << bin_file.c_str() << " ..."
                << std::endl;
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    uint64_t      fsize = reader.tellg();
    reader.seekg(0);

    load_bin_impl<T>(reader, fsize, data, npts, dim);
  }
  // load_bin functions END

  template<typename T>
  inline void load_bin(const std::string& bin_file, std::unique_ptr<T[]>& data,
                       _u64& npts, _u64& dim) {
    T* ptr;
    load_bin<T>(bin_file, ptr, npts, dim);
    data.reset(ptr);
  }

  template<typename T>
  inline void save_bin(const std::string& filename, T* data, _u64 npts,
                       _u64 ndims) {
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    grann::cout << "Writing bin: " << filename.c_str() << std::endl;
    int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
    writer.write((char*) &npts_i32, sizeof(int));
    writer.write((char*) &ndims_i32, sizeof(int));
    grann::cout << "bin: #pts = " << npts << ", #dims = " << ndims
                << ", size = " << npts * ndims * sizeof(T) + 2 * sizeof(int)
                << "B" << std::endl;

    //    data = new T[npts_u64 * ndims_u64];
    writer.write((char*) data, npts * ndims * sizeof(T));
    writer.close();
    grann::cout << "Finished writing bin." << std::endl;
  }



    template<typename T>
  inline void save_data_in_original_dimensions(const std::string& filename, T* base_data,
                                      _u64 npts, _u64 aligned_dim, _u64 original_dim) {

    if (original_dim > aligned_dim) {
      grann::cout<<"Error, original_dim must be at most aligned_dimension; NOT SAVING FILE." << std::endl;
      return;
    }
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    grann::cout << "Writing bin: " << filename.c_str() << std::endl;

    int npts_i32 = (int) npts, ndims_i32 = (int) original_dim;
    writer.write((char*) &npts_i32, sizeof(int));
    writer.write((char*) &ndims_i32, sizeof(int));
    grann::cout << "bin: #pts = " << npts_i32 << ", #dims = " << original_dim
                << ", size = "
                << (npts) * original_dim * sizeof(T) + 2 * sizeof(int)
                << "B" << std::endl;

    for (_u64 i = 0; i < npts; i++) {
      writer.write((char*) (base_data + (_u64) i * aligned_dim), original_dim * sizeof(T));
    }
    writer.close();
    grann::cout << "Finished writing bin." << std::endl;
  }


  template<typename T>
  inline void save_aligned_data_subset_in_orig_dimensions(const std::string& filename, T* base_data,
                                      _u64 npts, _u64 ndims, _u64 aligned_dim, 
                                      std::vector<_u32>& list_of_tags) {
    if (ndims > aligned_dim) {
      grann::cout<<"Error, dimension > aligned_dimension. Not saving" << std::endl;
      return;
    }
    bool valid = true;
    for (auto& x : list_of_tags) {
      if (x >= npts) {
        valid = false;
        break;
      }
    }

    if (list_of_tags.size() == 0)
      valid = false;

    if (!valid) {
      grann::cout
          << "Invalid list of ids to save. All entries must be between 0 and "
          << npts << std::endl;
    }

    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    grann::cout << "Writing bin: " << filename.c_str() << std::endl;

    int npts_i32 = (int) list_of_tags.size(), ndims_i32 = (int) ndims;
    writer.write((char*) &npts_i32, sizeof(int));
    writer.write((char*) &ndims_i32, sizeof(int));
    grann::cout << "bin: #pts = " << npts_i32 << ", #dims = " << ndims
                << ", size = "
                << (list_of_tags.size()) * ndims * sizeof(T) + 2 * sizeof(int)
                << "B" << std::endl;

    for (auto& id : list_of_tags) {
      writer.write((char*) (base_data + (_u64) id * aligned_dim), ndims * sizeof(T));
    }
    writer.close();
    grann::cout << "Finished writing bin." << std::endl;
  }



  // load_aligned_bin functions START

  template<typename T>
  inline void load_aligned_bin_impl(std::basic_istream<char>& reader,
                                    _u64 actual_file_size, T*& data, _u64& npts,
                                    _u64& dim, _u64& rounded_dim) {
    int npts_i32, dim_i32;
    reader.read((char*) &npts_i32, sizeof(int));
    reader.read((char*) &dim_i32, sizeof(int));
    npts = (unsigned) npts_i32;
    dim = (unsigned) dim_i32;

    _u64 expected_actual_file_size =
        npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
      std::stringstream stream;
      stream << "Error. File size mismatch. Actual size is " << actual_file_size
             << " while expected size is  " << expected_actual_file_size
             << " npts = " << npts << " dim = " << dim
             << " size of <T>= " << sizeof(T) << std::endl;
      grann::cout << stream.str() << std::endl;
      throw grann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
    }
    rounded_dim = ROUND_UP(dim, 8);
    grann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
                << ", aligned_dim = " << rounded_dim << "..." << std::flush;
    _u64 allocSize = npts * rounded_dim * sizeof(T);
    alloc_aligned(((void**) &data), allocSize, 8 * sizeof(T));

    for (_u64 i = 0; i < npts; i++) {
      reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
      memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    grann::cout << " done." << std::endl;
  }

  template<typename T>
  inline void load_aligned_bin(const std::string& bin_file, T*& data,
                               _u64& npts, _u64& dim, _u64& rounded_dim) {
    grann::cout << "Reading bin file " << bin_file << " ..." << std::flush;

    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    uint64_t      fsize = reader.tellg();
    reader.seekg(0);

    load_aligned_bin_impl(reader, fsize, data, npts, dim, rounded_dim);
  }

  template<typename InType, typename OutType>
  void convert_types(const InType* srcmat, OutType* destmat, _u64 npts,
                     _u64 dim) {
#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (_s64) npts; i++) {
      for (uint64_t j = 0; j < dim; j++) {
        destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
      }
    }
  }

  // this function will take in_file of n*d dimensions and save the output as a
  // floating point matrix
  // with n*(d+1) dimensions. All vectors are scaled by a large value M so that
  // the norms are <=1 and the final coordinate is set so that the resulting
  // norm (in d+1 coordinates) is equal to 1 this is a classical transformation
  // from MIPS to L2 search from "On Symmetric and Asymmetric LSHs for Inner
  // Product Search" by Neyshabur and Srebro

  template<typename T>
  float prepare_base_for_inner_products(const std::string in_file,
                                        const std::string out_file) {
    std::cout << "Pre-processing base file by adding extra coordinate"
              << std::endl;
    std::ifstream in_reader(in_file.c_str(), std::ios::binary);
    std::ofstream out_writer(out_file.c_str(), std::ios::binary);
    _u64          npts, in_dims, out_dims;
    float         max_norm = 0;

    _u32 npts32, dims32;
    in_reader.read((char*) &npts32, sizeof(uint32_t));
    in_reader.read((char*) &dims32, sizeof(uint32_t));

    npts = npts32;
    in_dims = dims32;
    out_dims = in_dims + 1;
    _u32 outdims32 = (_u32) out_dims;

    out_writer.write((char*) &npts32, sizeof(uint32_t));
    out_writer.write((char*) &outdims32, sizeof(uint32_t));

    _u64                 BLOCK_SIZE = 100000;
    _u64                 block_size = npts <= BLOCK_SIZE ? npts : BLOCK_SIZE;
    std::unique_ptr<T[]> in_block_data =
        std::make_unique<T[]>(block_size * in_dims);
    std::unique_ptr<float[]> out_block_data =
        std::make_unique<float[]>(block_size * out_dims);

    std::memset(out_block_data.get(), 0, sizeof(float) * block_size * out_dims);
    _u64 num_blocks = DIV_ROUND_UP(npts, block_size);

    std::vector<float> norms(npts, 0);

    for (_u64 b = 0; b < num_blocks; b++) {
      _u64 start_id = b * block_size;
      _u64 end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
      _u64 block_pts = end_id - start_id;
      in_reader.read((char*) in_block_data.get(),
                     block_pts * in_dims * sizeof(T));
      for (_u64 p = 0; p < block_pts; p++) {
        for (_u64 j = 0; j < in_dims; j++) {
          norms[start_id + p] +=
              in_block_data[p * in_dims + j] * in_block_data[p * in_dims + j];
        }
        max_norm =
            max_norm > norms[start_id + p] ? max_norm : norms[start_id + p];
      }
    }

    max_norm = std::sqrt(max_norm);

    in_reader.seekg(2 * sizeof(_u32), std::ios::beg);
    for (_u64 b = 0; b < num_blocks; b++) {
      _u64 start_id = b * block_size;
      _u64 end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
      _u64 block_pts = end_id - start_id;
      in_reader.read((char*) in_block_data.get(),
                     block_pts * in_dims * sizeof(T));
      for (_u64 p = 0; p < block_pts; p++) {
        for (_u64 j = 0; j < in_dims; j++) {
          out_block_data[p * out_dims + j] =
              in_block_data[p * in_dims + j] / max_norm;
        }
        float res = 1 - (norms[start_id + p] / (max_norm * max_norm));
        res = res <= 0 ? 0 : std::sqrt(res);
        out_block_data[p * out_dims + out_dims - 1] = res;
      }
      out_writer.write((char*) out_block_data.get(),
                       block_pts * out_dims * sizeof(float));
    }
    out_writer.close();
    return max_norm;
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector(const char* vec, _u64 vecsize) {
    _u64 max_prefetch_size = (vecsize / 64) * 64;
    for (_u64 d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char*) vec + d, _MM_HINT_T0);
  }

  // NOTE :: good efficiency when total_vec_size is integral multiple of 64
  inline void prefetch_vector_l2(const char* vec, _u64 vecsize) {
    _u64 max_prefetch_size = (vecsize / 64) * 64;
    for (_u64 d = 0; d < max_prefetch_size; d += 64)
      _mm_prefetch((const char*) vec + d, _MM_HINT_T1);
  }

};  // namespace grann

inline bool file_exists(const std::string& name) {
  struct stat buffer;
  auto        val = stat(name.c_str(), &buffer);
  grann::cout << " Stat(" << name.c_str() << ") returned: " << val << std::endl;
  return (val == 0);
}

inline _u64 get_file_size(const std::string& fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    _u64 end_pos = reader.tellg();
    grann::cout << " Tellg: " << reader.tellg() << " as u64: " << end_pos
                << std::endl;
    reader.close();
    return end_pos;
  } else {
    grann::cout << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

inline void wait_for_keystroke() {
  int a;
  std::cout << "Press any number to continue.." << std::endl;
  std::cin >> a;
}
