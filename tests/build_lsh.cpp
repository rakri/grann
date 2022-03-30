#include <string.h>
#include "lsh.h"
#include "utils.h"

template<typename T>
int build_lsh_index(const std::string &data_path, const std::string &labels_file, const _u32 num_tables,
                    const _u32 table_size, const std::string &save_path) {
  grann::Parameters params;
  params.Set<_u32>("num_tables", num_tables);
  params.Set<_u32>("table_size", table_size);

  grann::Metric     m = grann::Metric::L2;
  std::vector<_u32> idmap;

  grann::LSHIndex<T> lsh(m, data_path.c_str(), idmap, labels_file);
  lsh.build(params);
  lsh.save(save_path.c_str());

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 7 || argc > 8) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>] [data_file.bin] [filtered_index (0/1)] {labels_file (if filtered_index==1)} "
                 "[output_index_prefix]  "
              << "[num_tables] [table_size < 64] " << std::endl;
    exit(-1);
  }

  _u32 ctr = 2;

  const std::string data_path(argv[ctr++]);
  bool filtered_index = (bool) std::atoi(argv[ctr++]);
  std::string labels_file = "";
  if (filtered_index) {
	labels_file = std::string(argv[ctr++]);
  }

  const std::string save_path(argv[ctr++]);
  const _u32        num_tables = (_u32) atoi(argv[ctr++]);
  const _u32        table_size = (_u32) atoi(argv[ctr++]);

  if (std::string(argv[1]) == std::string("int8"))
    build_lsh_index<int8_t>(data_path, labels_file, num_tables, table_size, save_path);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_lsh_index<uint8_t>(data_path, labels_file, num_tables, table_size, save_path);
  else if (std::string(argv[1]) == std::string("float"))
    build_lsh_index<float>(data_path, labels_file, num_tables, table_size, save_path);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
