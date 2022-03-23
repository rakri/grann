#include "essentials.h"
#include "index.h"
#include "utils.h"
#include "lsh.h"
#include "math_utils.h"


#define SEARCH_MODE 0
#define BUILD_MODE 1


namespace grann {

	HashTable::HashTable(_u32 table_s, _u32 vector_d) {
		if (table_s > BITSET_MAX) {
			perror("Input table size is too large");
			exit(1);
		}

		random_hps.reserve(table_s);
		
		vector_dim = vector_d;
		table_size = table_s;
	}

	HashTable::~HashTable() {}

	void HashTable::generate_hps() {
		std::mt19937 rng(unsigned(std::time(0)));
		std::normal_distribution<float> gaussian_dist;

		for (size_t i = 0; i < table_size; i++) {
			std::vector<float> random_hp;
			std::mt19937 rng(unsigned(std::time(0)));
			random_hp.reserve(vector_dim);
			for (size_t j = 0; j < vector_dim; j++) {
				random_hp.push_back(gaussian_dist(rng));
			}
			add_hp(random_hp);
		}
	}

	std::vector<_u32> HashTable::get_bucket(bitstring bucket_id) {
		return hashed_vectors[(size_t) bucket_id.to_ulong()];	
	}

	template<typename T>
	bitstring HashTable::get_hash(const T *input_vector) {
		float dot_p = 0.0;
		bitstring input_bits;
		for (size_t i = 0; i < table_size; i++) {
			dot_p += std::inner_product(random_hps[i].begin(), random_hps[i].end(), input_vector, 0.0);
			if (dot_p > 0)
				input_bits[i] = 1;
			else
				input_bits[i] = 0;
		}
		return input_bits;
	}

	void HashTable::add_vector(bitstring vector_hash, _u32 vector_id) {
		hashed_vectors[(size_t) vector_hash.to_ulong()].push_back(vector_id);
	}

	void HashTable::add_hp(std::vector<float> hp) {
		random_hps.push_back(hp);
	}

	void HashTable::write_to_file(std::ofstream &out) {
		// 0. write null terminator to denote start of hashtable
		out.put('@');
		
		// 1. write hyperplane vectors
		for (const auto &hp : random_hps) {
			for (const auto &e : hp) {
				out.write(reinterpret_cast<const char *>(&e), sizeof(float));
			}
		}

		out.put('%');

		// 2. write map
		_u32 num_buckets = hashed_vectors.size();
		out.write(reinterpret_cast<const char *>(&num_buckets), sizeof(_u32));
		for (const auto &bucket : hashed_vectors) {
			size_t bucket_hash = (size_t) bucket.first;
			std::vector<_u32> bucket_contents = bucket.second;
			_u32 bucket_size = bucket_contents.size();
			out.write(reinterpret_cast<const char *>(&bucket_hash), sizeof(size_t));
			out.write(reinterpret_cast<const char *>(&bucket_size), sizeof(_u32));
			for (const auto &e : bucket_contents) {
				out.write(reinterpret_cast<const char *>(&e), sizeof(_u32));
			}
		}

		// 3. write ending char
		out.put('$');
	}

	
	template<typename T>
	LSHIndex<T>::LSHIndex(Metric m, const char *fname, 
												std::vector<_u32> &list_of_tags) 
							: ANNIndex<T>(m, fname, list_of_tags) {
		num_tables = 0;
		table_size = 0;
		vector_dim = 0;
	}

	template<typename T>
	LSHIndex<T>::LSHIndex(Metric m) : ANNIndex<T>(m) {
		num_tables = 0;
		table_size = 0;
		vector_dim = 0;
	}

	template<typename T>
	LSHIndex<T>::~LSHIndex() {}

	template<typename T>
	void LSHIndex<T>::build(Parameters &params) {
		num_tables = params.Get<_u32>("num_tables");
		table_size = params.Get<_u32>("table_size");
		tables.reserve(num_tables);
		for (size_t i = 0; i < num_tables; i++) {
			HashTable table = HashTable(table_size, this->_aligned_dim);
			table.generate_hps();
			tables.push_back(table);
		}

		for (auto &table : tables) {
			for (_s64 i = 0; i < (_s64) this->_num_points; i++) {
				const T *cur_vec = this->_data + (i * (_u64) this->_aligned_dim);
				bitstring cur_vec_hash = table.get_hash(cur_vec);
				table.add_vector(cur_vec_hash, this->_tag_map[i]);
			}
		}
	}

	template<typename T>
	_u32 LSHIndex<T>::search(const T *query, _u32 res_count, Parameters &search_params,
													 _u32 *indices, float *distances, QueryStats *stats) {
		float *query_float = new float[this->_aligned_dim];
		grann::convert_types(query, query_float, 1, this->_aligned_dim);

		std::vector<_u32> candidates;
		for (auto &table : tables) {
			bitstring query_hash = table.get_hash(query);
			std::vector<_u32> curr_bucket = table.get_bucket(query_hash);
			candidates.insert(candidates.end(), curr_bucket.begin(), curr_bucket.end());	
		}

		std::vector<Neighbor> best_candidates(res_count + 1);
		_u32									curr_size = 0;
		_u32 									max_size = res_count;
		_u32									cmps = 0;
		tsl::robin_set<_u32>	inserted;

		ANNIndex<T>::process_candidates_into_best_candidates_pool(query,
				candidates, best_candidates, max_size, curr_size, inserted, cmps);

		res_count = curr_size < res_count ? curr_size : res_count;
		for (_u32 i = 0; i < res_count; i++) {
      indices[i] = best_candidates[i].id;
      //    grann::cout<< indices[i] << "\t";
      if (distances != nullptr) {
        distances[i] = best_candidates[i].distance;
        //        grann::cout<< distances[i] << std::endl;
      }
    }
    if (stats != nullptr) {
      stats->n_cmps += cmps;
    }

		delete[] query_float;
		return res_count;
	}

	template<typename T>
	void LSHIndex<T>::save(const char *fname) {
		ANNIndex<T>::save_data_and_tags(fname);
		std::string index_file(fname);
		index_file += "_index.bin";
		std::cout << index_file << std::endl;

		std::ofstream out(index_file, std::ios::binary | std::ios::out);
		_u64 total_count = 0;

		out.put('*');

		// 1. write number of tables
		out.write(reinterpret_cast<const char *>(&num_tables), sizeof(_u32));
		out.write(reinterpret_cast<const char *>(&table_size), sizeof(_u32));

		// 2. write the tables
		for (auto &table : tables) {
			table.write_to_file(out);
			total_count++;
		}
		out.close();

		std::cout << "wrote " << total_count << "tables to disk" << std::endl;
	}

	template<typename T>
	void LSHIndex<T>::load(const char *fname) {
		ANNIndex<T>::load_data_and_tags(fname);
		std::string index_file(fname);
		index_file += "_index.bin";
		std::cout << index_file << std::endl;

		std::ifstream in(index_file, std::ios::binary | std::ios::in);

		char begin;
		in.get(begin);
		if ((begin) != '*') {
			printf("%d\n", begin);
			perror("Mistake in file formation, missing null terminator at begin. Exiting...");
			exit(1);
		}

		// 1. get number of tables and table_size
		_u32 f_num_tables, f_table_size;
		in.read(reinterpret_cast<char *>(&f_num_tables), sizeof(_u32));
		in.read(reinterpret_cast<char *>(&f_table_size), sizeof(_u32));

		// 2. read in table by table
		tables.clear();
		for (size_t i = 0; i < f_num_tables; i++) {
			// 1. verify starting character
			char start;
			in.get(start);
			if ((start) != '@') {
				printf("%d\n", start);
				perror("Mistake in file formation, missing null terminator at start. Exiting...");
				exit(1);
			}

			// 2. get random hyperplanes
			HashTable curr_table(f_table_size, this->_aligned_dim);
			for (size_t j = 0; j < f_table_size; j++) {
				std::vector<float> random_hp;
				for (size_t k = 0; k < this->_aligned_dim; k++) {
					float next_hp_element;
					in.read(reinterpret_cast<char *>(&next_hp_element), sizeof(float));
				}
				curr_table.add_hp(random_hp);
			}

			char mid;
			in.get(mid);
			if ((mid) != '%') {
				printf("%d\n", mid);
				perror("Mistake in file formation, missing null terminator at middle. Exiting...");
				exit(1);
			}
			
			// 3. get map
			_u32 f_num_buckets;
			in.read(reinterpret_cast<char *>(&f_num_buckets), sizeof(_u32));
			for (size_t j = 0; j < f_num_buckets; j++) {
				size_t f_bucket_hash_int;
				_u32 f_bucket_size;
				in.read(reinterpret_cast<char *>(&f_bucket_hash_int), sizeof(size_t));
				in.read(reinterpret_cast<char *>(&f_bucket_size), sizeof(_u32));
				bitstring f_bucket_hash_actual(f_bucket_hash_int);
				for (size_t k = 0; k < f_bucket_size; k++) {
					_u32 f_curr_id;
					in.read(reinterpret_cast<char *>(&f_curr_id), sizeof(_u32));
					curr_table.add_vector(f_bucket_hash_actual, f_curr_id);
				}
			}

			char end;
			in.get(end);
			if ((end) != '$') {
				printf("%d\n", end);
				perror("Mistake in file formation, missing null terminator at end. Exiting...");
				exit(1);
			}
			tables.push_back(curr_table);
		}

	}

	// EXPORTS
	template class LSHIndex<float>;
	template class LSHIndex<int8_t>;
	template class LSHIndex<uint8_t>;
} // namespace grann
