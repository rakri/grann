#!/usr/bin/env python3

#author: Siddharth Gollapudi
#email: t-sgollapudi@microsoft.com

import argparse
import itertools
import logging
import numpy as np

from base.distributions import Constant
from base.distributions import LogNormal
from base.distributions import Exponential
from base.distributions import Normal
from base.distributions import Uniform
from base.filter_class import FilterClass
from base.lib import first_n_primes
from typing import Any, Dict, List, Tuple


# logging init
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



########################
#  Filter-Class Setup  #
########################
# Change these to reflect the classes
'''
FILTER_CLASSES = [
                    FilterClass(3, Uniform([0, 3], 'filter_dist', 3), 'size class',
                                Constant(1, 'size_dist'), (1, 1)),
                    FilterClass(12, LogNormal([0.5, -1, 3.8], 'filter_dist', 12), 'color class',
                                Uniform([2, 6], 'size_dist', 4), (2, 6)),
                    FilterClass(3, Uniform([0, 3], 'filter_dist', 3), 'format class',
                                Constant(1, 'size_dist'), (1, 1)),
                    FilterClass(5, Exponential([0, 0.8], 'filter_dist', 5), 'date class',
                                Constant(1, 'size_dist'), (1, 1)),
                    FilterClass(4, Exponential([0, 0.4], 'filter_dist', 4), 'usage rights class',
                                Constant(1, 'size_dist'), (1, 1))
                ]
'''
FILTER_CLASSES = [
                    FilterClass(2000, Normal([1000,20], 'filter_dist', 2000), 'generic class',
                                Uniform([5, 20], 'size_dist', 15), (5, 20))
                 ]

_UNIVERSAL_LBL = -1

# don't touch this
def _get_total_num_filters() -> int:
    num_filters = 0
    for fc in FILTER_CLASSES:
        num_filters += fc.size
    return num_filters

_TOTAL_FILTERS = _get_total_num_filters()
_PRIMES = []



########################
#      Functions       #
########################
def _list_to_str(lst: List[Any]) -> str:
    '''
    Represents each list as a string:

    [a, b, c] -> "a,b,c"
    '''
    s = ",".join(map(str, lst)) + '\n'
    return s


def _tuple_hash(tup: Tuple[int, ...]) -> int:
    '''
    Converts a tuple of integers using a modulo/prime hash.
    '''
    M, R = 500 * (_TOTAL_FILTERS), 37
    hash = 0
    for t in tup:
        hash += (R * hash + t) % M
    return hash


def _primes_tuple_hash(tup: Tuple[int, ...]) -> int:
    prime_tup = np.zeros(len(tup))
    for i in range(len(tup)):
        prime_tup[i] = _PRIMES[tup[i]]
    return int(np.prod(prime_tup))


def build_parser() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate filter sets for each point in base set")
    # default output filename is "filter_assignments.txt"
    parser.add_argument('--fname', metavar='file_name', type=str, default='filter_assignments.txt',
                        help='name of file to output results to')
    # default is 1000000 points
    parser.add_argument('--np', metavar='num_points', type=int, default=100000,
                        help='number of points in base set')
    # default is unflattened
    parser.add_argument('--powerset', action='store_true',
                        help="If present, flatten the labels into tuples and store the powerset of \
                        the tuples")
    # default is unflattened
    parser.add_argument('--factor_hash', action='store_true',
                        help="If present, flatten the labels into tuples and store their factor_hashes")

    return vars(parser.parse_args())


def assign_filters_to_classes():
    '''
    Assigns each class a subset of the entire label "universe," based on
    the class' size.
    '''
    filters = list(range(1, _TOTAL_FILTERS + 1))
    for fc in FILTER_CLASSES:
        logger.info("Class {0} has {1} labels".format(fc.name, fc.size))
        fc.assign_elements(filters[:fc.size])
        del filters[:fc.size]


def generate_label_set() -> List[int]:
    '''
    Generates a label set. Refer to FilterClass/Distribution documentation
    for more information.
    '''
    label_set = []
    for fc in FILTER_CLASSES:
        sampled_elements = fc.sample_elements()
        if _UNIVERSAL_LBL > 0:
            if sampled_elements[0] == _UNIVERSAL_LBL:
                sampled_elements = [sampled_elements[0]]
            elif _UNIVERSAL_LBL in sampled_elements:
                sampled_elements = np.setdiff1d(sampled_elements, [_UNIVERSAL_LBL])
        label_set.extend(sampled_elements)
    return label_set


def generate_flattened_set_fh() -> List[int]:
    '''
    Generates a flattened label set, and hashes them into integers
    with factor hashing.

    1. Splits the label set by class
    2. Separates out labels from the same class into different sets
    3. Computes the factor hash for each set
    '''
    label_set = []
    for fc in FILTER_CLASSES:
        sampled_elements = fc.sample_elements()
        label_set.append(sampled_elements)
    label_set = list(itertools.product(*label_set))

    label_set = [_primes_tuple_hash(s) for s in label_set]
    return label_set


def generate_flattened_set_ps() -> List[Tuple[int, ...]]:
    '''
    Generates a flattened label set, along with the powerset.

    1. Splits up the label set by class
    2. Separates out labels from the same class into different sets
    3. Computes the powerset of all the different sets
    4. Recombines and removes duplicates
    '''
    label_set = []
    for fc in FILTER_CLASSES:
        sampled_elements = fc.sample_elements()
        label_set.append(sampled_elements)

    label_set = list(itertools.product(*label_set))
    temp = []
    for s in label_set:
        curr = list(itertools.chain.from_iterable(itertools.combinations(s, r)
                                             for r in range(1, len(s)+1)))
        start_len = len(curr)
        curr = set(map(_tuple_hash, curr))
        end_len = len(curr)
        if (start_len - end_len) > 0:
            logger.warning("{0} elements are duplicates".format(start_len - end_len))
        temp.extend(curr)

    return temp


def assign_label_sets(factor_hash: bool, powerset: bool, num_points: int) -> List[List[int]]:
    '''
    Assigns each point a label set, a list of randomly sampled labels.

    If flatten is true, then the label sets consist of tuples containing every possible
    combination of labels that could satisfy a query.
    '''
    label_sets = []
    if factor_hash:
        global _PRIMES
        logger.warning("Generating flattened label sets with factor hashing (might not work)")
        _PRIMES = first_n_primes(_TOTAL_FILTERS)
    elif powerset:
        logger.warning("Generating flattened label sets with powersets (very large)")

    for _ in range(num_points):
        label_set = []
        if powerset:
            label_set = generate_flattened_set_ps()
        elif factor_hash:
            label_set = generate_flattened_set_fh()
        else:
            label_set = generate_label_set()
        label_sets.append(label_set)
    return label_sets


def label_sets_to_file(label_sets: List[List[int]], output_fname: str):
    '''
    Writes label sets to disk
    '''
    output_file = open(output_fname, 'w+')

    for s in label_sets:
        str_label_set = _list_to_str(s)
        output_file.write(str_label_set)
    output_file.close()


def classes_label_gen_pipeline(factor_hash: bool, output_fname: str,
                               powerset: bool, num_points: int):
    '''
    Pipeline for generating labels for classes:
    1. assign subsets of the total labelset to each each class
    2. generate the labelset for each point
    3. write the labelsets to file
    '''
    # 1.
    logger.info("Assigning labels to classes")
    assign_filters_to_classes()
    logger.info("Finished assigning labels to classes")

    # 2.
    logger.info("Generating label sets")
    label_sets = assign_label_sets(factor_hash, powerset, num_points)
    logger.info("Finished generating label sets")

    # 3.
    logger.info("Writing label sets to disk")
    label_sets_to_file(label_sets, output_fname)
    logger.info("Done")


if __name__ == '__main__':
    args = build_parser()

    num_points = args['np']
    out_fname = args['fname']
    powerset = args['powerset']
    factor_hash = args['factor_hash']
    classes_label_gen_pipeline(factor_hash, out_fname, powerset, num_points)
