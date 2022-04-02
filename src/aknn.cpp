// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>

#include "utils.h"
#include "aknn.h"
#include "vamana.h"

namespace grann {

  // Initialize an vamana with metric m, load the data of type T with filename
  // (bin), and initialize num_points
  template<typename T>
  ApproxKNN<T>::ApproxKNN(Metric m, const char *filename,
                    std::vector<_u32> &list_of_tags,
                        std::string        labels_fname)
      : GraphIndex<T>(m, filename, list_of_tags, labels_fname) {
    grann::cout << "Initialized ApproxKNN Object with " << this->_num_points
                << " points, dim=" << this->_dim << "." << std::endl;
    _data_file_path = std::string(filename);
  }

  template<typename T>
  ApproxKNN<T>::ApproxKNN(Metric m) : GraphIndex<T>(m) {
    grann::cout << "Initialized Empty ApproxKNN Object." << std::endl;
  }

  // save the graph vamana on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T>
  void ApproxKNN<T>::save(const char *filename) {
    ANNIndex<T>::save_data_and_tags_and_labels(filename);

    if (this->_filtered_index) {
      if (this->_filter_to_medoid_id.size()) {
        std::ofstream medoid_writer(std::string(filename) +
                                    "_labels_to_medoids.txt");
        for (auto iter : this->_filter_to_medoid_id) {
          medoid_writer << iter.first << ", " << iter.second << std::endl;
          std::cout << iter.first << ", " << iter.second << std::endl;
        }
        medoid_writer.close();  
    }
    }

    long long     total_gr_edges = 0;
    _u64          vamana_size = 0;
    std::ofstream out(std::string(filename), std::ios::binary | std::ios::out);

    out.write((char *) &vamana_size, sizeof(uint64_t));
    out.write((char *) &this->_max_degree, sizeof(unsigned));
    out.write((char *) &this->_start_node, sizeof(unsigned));
    for (unsigned i = 0; i < this->_num_points; i++) {
      unsigned GK = (unsigned) this->_out_nbrs[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) this->_out_nbrs[i].data(), GK * sizeof(unsigned));
      total_gr_edges += GK;
    }
    vamana_size = out.tellp();
    out.seekp(0, std::ios::beg);
    out.write((char *) &vamana_size, sizeof(uint64_t));
    out.close();
  }

  // load the vamana from file and update the width (max_degree), ep
  // (navigating node id), and _out_nbrs (adjacency list)
  template<typename T>
  void ApproxKNN<T>::load(const char *filename) {
    ANNIndex<T>::load_data_and_tags_and_labels(filename);

      std::string labels_to_medoids_file = std::string(filename) + "_labels_to_medoids.txt";
      if (file_exists(labels_to_medoids_file)) {
        std::ifstream medoid_stream(labels_to_medoids_file);

        std::string line, token;
        unsigned    line_cnt = 0;

        _filter_to_medoid_id.clear();

        while (std::getline(medoid_stream, line)) {
          std::istringstream iss(line);
          _u32               cnt = 0;
          _u32               medoid = 0;
          std::string        label;
          while (std::getline(iss, token, ',')) {
            token.erase(std::remove(token.begin(), token.end(), '\n'),
                        token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'),
                        token.end());
            if (cnt == 0)
              label = token;
            else
              medoid = (_u32) stoul(token);
            cnt++;
          }
          _filter_to_medoid_id[label] = medoid;
          line_cnt++;
        }
      }


    std::ifstream in(filename, std::ios::binary);
    _u64          expected_file_size;
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &this->_max_degree, sizeof(unsigned));
    in.read((char *) &this->_start_node, sizeof(unsigned));
    grann::cout << "Loading vamana index " << filename << "..." << std::flush;

    _u64     cc = 0;
    unsigned nodes = 0;
    while (!in.eof()) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (in.eof())
        break;
      cc += k;
      ++nodes;
      std::vector<unsigned> tmp(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));

      this->_out_nbrs.emplace_back(tmp);
      if (nodes % 10000000 == 0)
        grann::cout << "." << std::flush;
    }
    if (this->_out_nbrs.size() != this->_num_points) {
      grann::cout << "ERROR. mismatch in number of points. Graph has "
                  << this->_out_nbrs.size() << " points and loaded dataset has "
                  << this->_num_points << " points. " << std::endl;
      return;
    }

    grann::cout << "..done. ApproxKNN has " << nodes << " nodes and " << cc
                << " out-edges" << std::endl;
  }

  /**************************************************************
   *      Support for Static ApproxKNN Building and Searching
   **************************************************************/



  template<typename T>
  void ApproxKNN<T>::build(Parameters &build_parameters) {
    grann::Timer build_timer;


    unsigned num_threads = build_parameters.Get<unsigned>("num_threads");
    unsigned L = build_parameters.Get<unsigned>("L");
    unsigned degree_bound = build_parameters.Get<unsigned>("R");
    float    alpha = build_parameters.Get<float>("alpha");

    grann::cout << "Starting vamana build with listSize L=" << L
                << ", degree bound R=" << degree_bound
                << ", and alpha=" << alpha << std::endl;

    std::vector<_u32> dummy_idmap;
    _vamana_for_build = new Vamana<T>(this->_metric, _data_file_path.c_str(), dummy_idmap);
    _vamana_for_build->build(build_parameters);

    this->_out_nbrs.resize(this->_num_points);
    for (auto &x : this->_out_nbrs)
      x.reserve(1.05 * degree_bound);

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    _u32             progress_milestone = (_u32)(this->_num_points / 10);
    std::atomic<int> milestone_marker{0};

#pragma omp parallel for schedule(static, 64)
    for (_u32 location = 0; location < this->_num_points; location++) {
      if (location % progress_milestone == 0) {
        ++milestone_marker;

        std::stringstream msg;
        msg << (milestone_marker * 10) << "\% of build completed \r";
        grann::cout << msg.str();
      }

      std::vector<_u32> res_ids;
      std::vector<float> res_dists;
      res_ids.resize(degree_bound);
      res_dists.resize(degree_bound);
    
      _u32 res_count = _vamana_for_build->search(this->_data + (_u64) location * this->_aligned_dim, degree_bound, build_parameters,
                    res_ids.data(),
                    res_dists.data());

      for (_u32 i = 0; i < res_count; i++)
          this->_out_nbrs[location].emplace_back(res_ids[i]);
      
    }

    grann::cout << "done." << std::endl;
    this->_has_built = true;
    this->update_degree_stats();

   delete[] _vamana_for_build;
    grann::cout << "Total build time: "
                << ((double) build_timer.elapsed() / (double) 1000000) << "s"
                << std::endl;
  }

  template<typename T>
  _u32 ApproxKNN<T>::search(const T *query, _u32 res_count,
                         Parameters &search_params, _u32 *indices,
                         float *distances, QueryStats *stats,
												 std::vector<label> search_filters) {
    _u32                     search_list_size = search_params.Get<_u32>("L");
    std::vector<unsigned>    init_ids;
    tsl::robin_set<unsigned> visited(10 * search_list_size);
    std::vector<Neighbor>    top_candidate_list, expanded_nodes_info;
    tsl::robin_set<unsigned> expanded_nodes_ids;

    if (search_filters.size()!=0) {
      for (auto &x : search_filters)
    if (_filter_to_medoid_id.find(x) != _filter_to_medoid_id.end())
      init_ids.emplace_back(_filter_to_medoid_id[x]);
    } else
    init_ids.emplace_back(this->_start_node);

    auto algo_fetched_count = this->greedy_search_to_fixed_point(
        query, search_list_size, init_ids, expanded_nodes_info,
        expanded_nodes_ids, top_candidate_list, search_filters, stats);

    //     _u64 pos = 0;
    for (_u32 i = 0; i < res_count; i++) {
      if (i >= res_count)
        break;
      indices[i] = this->_tag_map[top_candidate_list[i].id];
      distances[i] = top_candidate_list[i].distance;
    }
    return std::min(res_count, algo_fetched_count);
  }

  // EXPORTS
  template class ApproxKNN<float>;
  template class ApproxKNN<int8_t>;
  template class ApproxKNN<uint8_t>;
}  // namespace grann
