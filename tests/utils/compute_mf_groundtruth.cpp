#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <string.h>
#include <atomic>
#include <cstring>
#include <iomanip>
#include <set>
#include "utils.h"
#include "index.h"
#include "../src/aux_compute_groundtruth.cpp"
#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#else
#include <Windows.h>
#endif
inline void parse_label_string(std::string               label_string,
                               std::vector<std::string> &labels) {
  std::istringstream iss(label_string);
  // std::cout<<label_string << std::endl;
  labels.clear();
  std::string token;
  while (getline(iss, token, ',')) {
    token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
    token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
    labels.push_back(token);
  }
}
template<typename T>
void compute_new_groundtruth(int gd_argc, char **gd_argv,
                             unsigned basefile_gd_argc, unsigned tru_gd_argc,
                             std::string            remapped_tru_bin,
                             std::vector<unsigned> &pts_in_index);
template<typename T>
void compute_filtered_gt(
    const std::string data_type, const std::string &coord_file,
    const std::string &map_file, const std::string filter_label,
    const std::string &query_file, const _u64 true_nn_count,
    const std::string &out_groundtruth_file, const bool use_universal_label,
    const std::string universal_label);
int main(int argc, char **argv) {
  if (argc != 8 && argc != 9) {
    std::cout
        << "Usage:\n"
        << argv[0] << " [data_type] [coordinates_file.bin] [mappings_file.tsv]"
        << " [filter_label(s) {comma separated for OR queries}] "
           "[query_file.bin]  "
           " [K/nbr_count] [output_filtered_gt] [universal_label (optional)]\n";
    exit(-1);
  }
  const std::string data_type(argv[1]);
  const std::string coord_file(argv[2]);
  const std::string map_file(argv[3]);
  const std::string filter_string(argv[4]);
  const std::string query_file(argv[5]);
  const _u64        true_nn_count = (_u64) atoi(argv[6]);
  const std::string out_groundtruth_file(argv[7]);
  std::string       universal_label = "";
  bool              use_universal_label = false;
  if (argc == 9) {
    universal_label = std::string(argv[8]);
    use_universal_label = true;
  }
  if (std::string(argv[1]) == std::string("int8"))
    compute_filtered_gt<int8_t>(data_type, coord_file, map_file, filter_string,
                                query_file, true_nn_count, out_groundtruth_file,
                                use_universal_label, universal_label);
  else if (std::string(argv[1]) == std::string("uint8"))
    compute_filtered_gt<uint8_t>(data_type, coord_file, map_file, filter_string,
                                 query_file, true_nn_count,
                                 out_groundtruth_file, use_universal_label,
                                 universal_label);
  else if (std::string(argv[1]) == std::string("float"))
    compute_filtered_gt<float>(data_type, coord_file, map_file, filter_string,
                               query_file, true_nn_count, out_groundtruth_file,
                               use_universal_label, universal_label);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
template<typename T>
void compute_filtered_gt(
    const std::string data_type, const std::string &coord_file,
    const std::string &map_file, const std::string filter_string,
    const std::string &query_file, const _u64 true_nn_count,
    const std::string &out_groundtruth_file, const bool use_universal_label,
    const std::string universal_label) {
  auto                          s = std::chrono::high_resolution_clock::now();
  auto                          p_time = s;
  auto                          c_time = s;
  std::chrono::duration<double> diff = c_time - p_time;
  p_time = std::chrono::high_resolution_clock::now();
  std::ifstream coord_stream(coord_file, std::ios::binary);
  std::ifstream quer_stream(query_file, std::ios::binary);
  std::ifstream map_stream(map_file);
  _u64              nn_cnt = true_nn_count;
  const std::string filtered_base(out_groundtruth_file + "_filtered_base.bin");
  const std::string dummy_mapped_gnd_truth_file(out_groundtruth_file +
                                                "_dummy");
  const std::string filtered_tru_remap(out_groundtruth_file);
  const std::string gd_progname(dummy_mapped_gnd_truth_file + "_dummy_prog");
  int               gd_argc = 6;
  char **           gd_argv = new char *[6];
  unsigned          progname_gd_argc = 0, datatype_gd_argc = 1;
  unsigned          basefile_gd_argc = 2, quer_gd_argc = 3;
  unsigned          recAt_gd_argc = 4, tru_gd_argc = 5;
  gd_argv[progname_gd_argc] = const_cast<char *>(gd_progname.c_str());
  gd_argv[datatype_gd_argc] = const_cast<char *>(data_type.c_str());
  gd_argv[basefile_gd_argc] = const_cast<char *>(filtered_base.c_str());
  gd_argv[quer_gd_argc] = const_cast<char *>(query_file.c_str());
  std::string rec_at_str = std::to_string(nn_cnt);
  gd_argv[recAt_gd_argc] = const_cast<char *>(rec_at_str.c_str());
  gd_argv[tru_gd_argc] =
      const_cast<char *>(dummy_mapped_gnd_truth_file.c_str());
  std::ofstream out_coord_stream = std::ofstream(filtered_base);
  std::cout << "Loading base file " << coord_file << "...\n";
  unsigned num_pts, num_dims;
  unsigned num_filt_pts = 0;
  coord_stream.read((char *) &num_pts, sizeof(num_pts));
  coord_stream.read((char *) &num_dims, sizeof(num_dims));

	// convert comma-separated filters from cmd line to a vector
  std::vector<std::string> parsed_filters;
  parse_label_string(filter_string, parsed_filters);
	std::sort(parsed_filters.begin(), parsed_filters.end());
  out_coord_stream.write((char *) &num_filt_pts, sizeof(num_filt_pts));
  out_coord_stream.write((char *) &num_dims, sizeof(num_dims));
  std::vector<T>        pt(num_dims);
  std::vector<unsigned> filtered_pts_index(0);
  std::string           line, token;
  unsigned              line_cnt = 0;
	
	// if subset query is satisfied, add base point for consideration
  while (std::getline(map_stream, line)) {
    std::istringstream iss(line);
    coord_stream.read((char *) pt.data(), sizeof(T) * num_dims);
		std::vector<grann::label> line_labels;
		parse_label_string(line, line_labels);
		std::sort(line_labels.begin(), line_labels.end());
		if (std::includes(line_labels.begin(), line_labels.end(), parsed_filters.begin(), parsed_filters.end())) {
			filtered_pts_index.push_back(line_cnt);
      out_coord_stream.write((char *) pt.data(), sizeof(T) * num_dims);
      num_filt_pts++;
		}
    line_cnt++;
  }
  if (num_pts != line_cnt) {
    std::cout
        << "Error: Number of labels (ignoring unlabeled errors)- expected:"
        << ", found:" << line_cnt << ")\n";
  }
  out_coord_stream.seekp(0, std::ios::beg);
  out_coord_stream.write((char *) &num_filt_pts, sizeof(num_filt_pts));
  out_coord_stream.close();
  coord_stream.close();
  map_stream.close();
  c_time = std::chrono::high_resolution_clock::now();
  diff = c_time - p_time;
  std::cout << "Base Vectors Filtering time: " << diff.count() << "\n"
            << "Starting ground truth computation\n";
  T *    full_data;
  size_t tmp_num_pts, tmp_num_dims;
  grann::load_bin<T>(coord_file, full_data, tmp_num_pts, tmp_num_dims);
  compute_new_groundtruth<T>(gd_argc, gd_argv, basefile_gd_argc, tru_gd_argc,
                             filtered_tru_remap, filtered_pts_index);
  p_time = s;
  c_time = std::chrono::high_resolution_clock::now();
  diff = c_time - p_time;
  std::cout << "Total Filtering time: " << diff.count() << "\n";
  std::remove(filtered_base.c_str());
  std::remove(dummy_mapped_gnd_truth_file.c_str());
  return;
}
template<typename T>
void compute_new_groundtruth(int gd_argc, char **gd_argv,
                             unsigned basefile_gd_argc, unsigned tru_gd_argc,
                             std::string            remapped_tru_bin,
                             std::vector<unsigned> &pts_in_index) {
  const std::string post_del_base(gd_argv[basefile_gd_argc]);
  const std::string post_del_truth(gd_argv[tru_gd_argc]);
  unsigned          num_pts_to_save = pts_in_index.size();
  std::cout << "Going to calculate ground truth over " << num_pts_to_save
            << " filtered points" << std::endl;
  aux_main<T>(gd_argc, gd_argv);
  std::ifstream in(post_del_truth, std::ios::binary);
  std::ofstream out_rm(remapped_tru_bin, std::ios::binary | std::ios::out);
  unsigned      gd_quers, gd_rec_at;
  in.read((char *) &gd_quers, sizeof gd_quers);
  out_rm.write((char *) &gd_quers, sizeof gd_quers);
  in.read((char *) &gd_rec_at, sizeof gd_rec_at);
  out_rm.write((char *) &gd_rec_at, sizeof gd_rec_at);
  std::vector<unsigned> quer_nns(gd_rec_at, 0);
  for (unsigned q_idx = 0; q_idx < gd_quers; q_idx++) {
    in.read((char *) quer_nns.data(), sizeof(unsigned) * gd_rec_at);
    for (unsigned nn_rnk = 0; nn_rnk < gd_rec_at; nn_rnk++) {
      quer_nns[nn_rnk] = pts_in_index[quer_nns[nn_rnk]];
    }
    out_rm.write((char *) quer_nns.data(), sizeof(unsigned) * gd_rec_at);
  }
  std::vector<T> quer_dists(gd_rec_at, 0);
  for (unsigned q_idx = 0; q_idx < gd_quers; q_idx++) {
    in.read((char *) quer_dists.data(), sizeof(float) * gd_rec_at);
    out_rm.write((char *) quer_dists.data(), sizeof(float) * gd_rec_at);
  }
  in.close();
  out_rm.close();
  std::cout << "\nNew Ground truth computed and saved\n";
  return;
}
