#include <string.h>
#include "lsh.h"
#include "utils.h"

template<typename T>
int build_lsh_index(const std::string &data_path, const _u32 num_tables,
                    const _u32 table_size, const std::string &save_path) {
  grann::Parameters params;
  params.Set<_u32>("num_tables", num_tables);
  params.Set<_u32>("table_size", table_size);

  grann::Metric     m = grann::Metric::L2;
  std::vector<_u32> idmap;

  grann::LSHIndex<T> lsh(m, data_path.c_str(), idmap);
  lsh.build(params);
  lsh.save(save_path.c_str());

  return 0;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0]
              << "  [data_type<int8/uint8/float>] [data_file.bin]  "
                 "[output_index_prefix]  "
              << "[num_tables] [table_size < 64] " << std::endl;
    exit(-1);
  }

  _u32 ctr = 2;

  const std::string data_path(argv[ctr++]);
  const std::string save_path(argv[ctr++]);
  const _u32        num_tables = (_u32) atoi(argv[ctr++]);
  const _u32        table_size = (_u32) atoi(argv[ctr++]);

  if (std::string(argv[1]) == std::string("int8"))
    build_lsh_index<int8_t>(data_path, num_tables, table_size, save_path);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_lsh_index<uint8_t>(data_path, num_tables, table_size, save_path);
  else if (std::string(argv[1]) == std::string("float"))
    build_lsh_index<float>(data_path, num_tables, table_size, save_path);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
