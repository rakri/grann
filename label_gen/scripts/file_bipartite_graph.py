#!/usr/bin/env python3

import argparse
import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
from typing import Any, Dict

LABELS_FILE = '/home/siddharth/Downloads/labels_files/labels500_sift10k_norm_unif5_25.txt'
NUM_PTS = 10000
p_hp = 0.3

def build_parser() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate filter sets for each point in base set")
    # no default
    parser.add_argument('--labels_file', metavar='file_name', type=str,
                        help='full path to file containing comma-separated labels on each line')
    # default is 1000000 points
    parser.add_argument('--np', metavar='num_points', type=int, default=1000000,
                        help='number of points in base set')
    # default is 1.0 (i.e. no effect)
    parser.add_argument('--hp', metavar="include_hyperparameter", type=float, default=1.0,
                        help="parameter (>0) to influence probability with which edges are added. higher -> smaller buckets")
    # no default
    parser.add_argument('--out_file', metavar='output_filename', type=str,
                        help='full path to the output file (to be created)')

    return vars(parser.parse_args())


def create_bipartite_graph(num_pts: int, labels_file: str):
    pts_to_labels = {}
    pts = list(range(num_pts))
    labels = set()

    print('Creating bipartite graph... ', end='')
    with open(labels_file, 'r+') as fd:
        i = 0
        for line in fd:
            if i == num_pts:
                break

            curr_labels = line.rstrip().split(',')[:5]
            pts_to_labels[i] = curr_labels
            labels.update(curr_labels)
            i += 1

    bipartite_edges = [(a, b) for a, i in pts_to_labels.items() for b in i]

    B = nx.Graph()
    B.add_nodes_from(pts, bipartite=0)
    B.add_nodes_from(labels, bipartite=1)
    B.add_edges_from(bipartite_edges)
    print('done.')
    return B


def create_hash_graph(b_graph, p_hp: int):
    print('Creating label buckets...', end='')
    labels = {n for n, d in b_graph.nodes(data=True) if d["bipartite"] == 1}
    H = nx.Graph()
    H.add_nodes_from(labels)
    for lbl_node in labels:
        lbl_node_nbrs = [nbr for nbr in b_graph.neighbors(lbl_node)]
        lbl_node_nbrs_nbrs = set([label for nbr in lbl_node_nbrs for label in b_graph.neighbors(nbr)])
        curr_p = 1 / (p_hp * len(lbl_node_nbrs_nbrs))
        for elem in set(lbl_node_nbrs_nbrs):
            flip = np.random.binomial(1, curr_p)
            if not flip:
                lbl_node_nbrs_nbrs.remove(elem)
        edges_to_add = [(lbl_node, b) for b in lbl_node_nbrs_nbrs]
        H.add_edges_from(edges_to_add)
    print('done.')
    print('There are', nx.number_connected_components(H), 'buckets.')
    return H


def create_label_hashes(hash_graph, out_file: str):
    S = [hash_graph.subgraph(c).copy() for c in nx.connected_components(hash_graph)]
    with open(out_file, 'w+') as fd:
        for ind, sg in enumerate(S):
            for n in sg.nodes:
                fd.write(n)
                fd.write(" ")
                fd.write(str(ind + 1))
                fd.write('\n')


if __name__ == '__main__':
    args = build_parser()
    labels_file = args['labels_file']
    num_pts = args['np']
    p_hp = args['hp']
    out_file = args['out_file']

    bi_graph = create_bipartite_graph(num_pts, labels_file)
    hash_graph = create_hash_graph(bi_graph, p_hp)
    create_label_hashes(hash_graph, out_file)
