#!/usr/bin/env python3

#author: Siddharth Gollapudi
#email: t-sgollapudi@microsoft.com

import argparse

from typing import Any, Dict, List, Tuple


def build_parser() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="derive statistics for labels in labels file")
    parser.add_argument('-f', metavar='file_name', type=str, default='',
                        help='path of input file')
    parser.add_argument('-u', metavar='universal_label', type=str, default=-1,
                        help='if used, provide universal label to correctly provide stats')
    return vars(parser.parse_args())


def read_file(fname: str) -> Dict[str, int]:
    counts = {}
    with open(fname) as file:
        for line in file:
            line = line.rstrip()
            for label in line.split(','):
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1
    return counts


def universal_corrector(counts: Dict[str, int], univ_lbl: str) -> List[Tuple[str, int]]:
    corrected_counts = counts
    if univ_lbl != -1:
        univ_lbl_count = corrected_counts[univ_lbl]
        del corrected_counts[univ_lbl]
        for label in corrected_counts:
            corrected_counts[label] += univ_lbl_count
    corrected_counts = list(sorted(counts.items(), key=lambda item: item[1]))
    return corrected_counts


def get_label_percentiles(label_counts: List[Tuple[str, int]]):
    num_counts = len(label_counts)
    print(num_counts, int(num_counts * (1/4)), int(num_counts * (2/4)), int(num_counts * (3/4)))
    min_label, max_label = label_counts[0], label_counts[-1]
    per25_label = label_counts[int(num_counts * (1/4))]
    per50_label = label_counts[int(num_counts * (2/4))]
    per75_label = label_counts[int(num_counts * (3/4))]
    print("least common label: {0}, with {1} occurrences".format(min_label[0], min_label[1]))
    print("25th percentile label: {0}, with {1} occurrences".format(per25_label[0], per25_label[1]))
    print("50th percentile label: {0}, with {1} occurrences".format(per50_label[0], per50_label[1]))
    print("75th percentile label: {0}, with {1} occurrences".format(per75_label[0], per75_label[1]))
    print("most common label: {0}, with {1} occurrences".format(max_label[0], max_label[1]))


if __name__ == '__main__':
    args = build_parser()
    counts = read_file(args['f'])
    print(counts)
    counts = universal_corrector(counts, args['u'])
    get_label_percentiles(counts)
