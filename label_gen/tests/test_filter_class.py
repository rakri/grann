#!/usr/bin/env python3

#author: Siddharth Gollapudi
#email: t-sgollapudi@microsoft.com

import numpy as np
import pytest

from base.distributions import Constant, Uniform
from base.filter_class import FilterClass


def test_filter_class_construction():
    test_size = 5
    test_dist = Uniform([0, 5], 'test_filter_dist', 5)
    test_name = 'test class'
    test_dist2 = Constant(1, 'test_size_dist')
    test_size_range = (1, 1)

    test_class = FilterClass(test_size, test_dist, test_name,
                                test_dist2, test_size_range)

    assert test_class


def test_filter_class_inputs():
    test_size = 5
    test_dist = Uniform([0, 5], 'test_filter_dist', 5)
    test_name = 'test class'
    test_dist2 = Constant(1, 'test_size_dist')
    test_size_range = (1, 1)

    test_class = FilterClass(test_size, test_dist, test_name,
                                test_dist2, test_size_range)

    assert test_size == test_class.size
    assert test_dist.name == test_class._filter_dist.name
    assert test_name == test_class.name
    assert test_dist2.name == test_class._sample_size_dist.name
    assert test_size_range == test_class._sample_size_range


def test_filter_class_element_assigning_success():
    test_size = 5
    test_dist = Uniform([0, 5], 'test_filter_dist', 5)
    test_name = 'test class'
    test_dist2 = Constant(1, 'test_size_dist')
    test_size_range = (1, 1)

    test_class = FilterClass(test_size, test_dist, test_name,
                                test_dist2, test_size_range)

    test_elems = [1, 2, 3, 4, 5]
    test_class.assign_elements(test_elems)

    assert test_elems == test_class._elements


def test_filter_class_element_assigning_failure():
    test_size = 5
    test_dist = Uniform([0, 5], 'test_filter_dist', 5)
    test_name = 'test class'
    test_dist2 = Constant(1, 'test_size_dist')
    test_size_range = (1, 1)

    test_class = FilterClass(test_size, test_dist, test_name,
                                test_dist2, test_size_range)

    test_elems = [1, 2, 3, 4]
    with pytest.raises(ValueError) as excinfo:
        test_class.assign_elements(test_elems)

        assert "Assigned incorrect number of elements" == str(excinfo.value)


def test_filter_class_sampling_success():
    test_size = 5
    test_dist = Uniform([0, 5], 'test_filter_dist', test_size)
    test_name = 'test class'
    test_dist2 = Uniform([0, 2], 'test_size_dist', 2)
    test_size_range = (1, 3)

    test_class = FilterClass(test_size, test_dist, test_name,
                                test_dist2, test_size_range)

    test_class.assign_elements(list(range(test_size)))
    test_sample = test_class.sample_elements()
    assert len(test_sample) == 1 or len(test_sample) == 2
    assert all(s in range(test_size) for s in test_sample)


def test_filter_class_sampling_fail():
    test_size = 5
    test_dist = Uniform([0, 5], 'test_filter_dist', test_size)
    test_name = 'test class'
    test_dist2 = Uniform([0, 2], 'test_size_dist', 2)
    test_size_range = (1, 1)

    test_class = FilterClass(test_size, test_dist, test_name,
                                test_dist2, test_size_range)

    with pytest.raises(ValueError) as excinfo:
        test_sample = test_class.sample_elements()

        assert "Length of choices must equal distribution size" == str(excinfo.value)
