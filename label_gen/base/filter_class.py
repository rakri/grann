#!/usr/bin/env python3

#author: Siddharth Gollapudi
#email: t-sgollapudi@microsoft.com

import numpy as np

from base.distribution import Distribution
from numpy.typing import NDArray
from typing import List, Tuple



class FilterClass:
    """
    Class representing a fliter class.

    A good example would be for image search, where some of the filter
    classes include size, color, and format.
    """
    def __init__(self, class_size: int, filter_dist: Distribution,
                 name: str, sample_size_dist: Distribution, sample_size_range: Tuple[int, int]):
        self._sample_size_dist = sample_size_dist
        self._filter_dist = filter_dist
        self._class_size = class_size
        self._name = name
        self._sample_size_range = sample_size_range
        self._elements = []


    @property
    def size(self) -> int:
        return self._class_size


    @property
    def name(self) -> str:
        return self._name


    def assign_elements(self, elements: List[int]) -> None:
        '''
        Sets the elements of the class to a given list of integers.

        Note that other label types are expected to be hashed externally.
        '''
        if len(elements) != self.size:
            raise ValueError("Assigned incorrect number of elements")
        self._elements = elements


    def sample_elements(self) -> NDArray:
        '''
        Using internal attributes, samples some number of elements from the class.

        The number of elements is randomly distributed, as is the elements themselves.
        '''
        possible_sizes = list(range(*self._sample_size_range))
        num_samples = self._sample_size_dist.draw_samples(possible_sizes, 1)[0]

        if not self._elements:
            raise ValueError("Elements need to be assigned for this class before sampling.")

        samples = self._filter_dist.draw_samples(self._elements, num_samples)
        return samples
