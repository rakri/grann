#!/usr/bin/env python3

#author: Siddharth Gollapudi
#email: t-sgollapudi@microsoft.com

import numpy as np
import pytest

from base.distributions import Constant
from base.distributions import Exponential
from base.distributions import Uniform



def test_distribution_construction():
    test_params = [0, 1]
    test_name = "test expon probs"
    test_size = 10

    test_distr = Exponential(test_params, test_name, test_size)

    assert test_distr


def test_distribution_inputs():
    test_params = [0, 1]
    test_name = "test expon probs"
    test_size = 10

    test_distr = Exponential(test_params, test_name, test_size)

    assert test_distr.params == test_params
    assert test_distr.name == test_name
    assert test_distr.size == test_size


def test_uniform_probabilities():
    baseline_probs = np.ones(10) / 10

    test_params = [0]
    test_name = "test uniform probs"
    test_size  = 10

    test_distr = Uniform(test_params, test_name, test_size)
    assert np.array_equal(test_distr.probabilities, baseline_probs)


def test_nonuniform_probabilities():
    baseline_probs = [0.39349879439218544,0.3834291999414516,0.14105571980327433,
                      0.051891499375264216,0.019089815791720442,0.007022750765523879,0.0025835256271072088,
                      0.0009504259639523655,0.00034964217249356,0.0001286261670269677]

    test_params = [0, 1]
    test_name = "test expon probs"
    test_size = 10

    test_distr = Exponential(test_params, test_name, test_size)
    assert np.array_equal(test_distr.probabilities, baseline_probs)


def test_nonconstant_sample_drawing():
    test_params = [0, 1]
    test_name = "test expon probs"
    test_size = 10

    test_distr = Exponential(test_params, test_name, test_size)
    test_samples = test_distr.draw_samples(list(range(10)), 3)

    assert len(test_samples) == 3


def test_constant_sample_drawing():
    test_params = 5
    test_name = "test constant sampling"

    test_distr = Constant(test_params, test_name)
    test_samples = test_distr.draw_samples([], 4)

    assert all(x == 5 for x in test_samples)
    assert len(test_samples) == 4
