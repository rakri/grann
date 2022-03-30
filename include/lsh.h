#pragma once

#include "utils.h"
#include "distance.h"
#include "percentile_stats.h"
#include "index.h"

namespace grann {

  const int                       BITSET_MAX = 64;
  typedef std::bitset<BITSET_MAX> bitstring;

  class HashTable {
   public:
    HashTable(_u32 table_size, _u32 vector_dim);
    ~HashTable();

    void generate_hps();

    std::vector<_u32> get_bucket(bitstring bucket_id);

    template<typename T>
    bitstring get_hash(const T *input_vector);

    void add_vector(bitstring vector_hash, _u32 vector_id);
    void add_hp(std::vector<float> hp);

    void write_to_file(std::ofstream &out);
    void read_from_file(std::ifstream &in);

   protected:
    _u32 vector_dim;  // dimension of points stored/each hp vector
    _u32 table_size;  // number of hyperplanes
    // float **random_hps;
    std::vector<std::vector<float>>     random_hps;
    std::map<size_t, std::vector<_u32>> hashed_vectors;
  };

  template<typename T>
  class LSHIndex : public ANNIndex<T> {
   public:
    LSHIndex(Metric m, const char *filename, std::vector<_u32> &list_of_tags);
    LSHIndex(Metric m);
    ~LSHIndex();

    void save(const char *filename);
    void load(const char *filename);

    void build(Parameters &parameters);

    _u32 search(const T *query, _u32 res_count, Parameters &search_params,
                _u32 *indices, float *distances, QueryStats *stats = nullptr);

   protected:
    _u32                   num_tables;
    _u32                   table_size;
    _u32                   vector_dim;
    std::vector<HashTable> tables;
  };

}  // namespace grann
