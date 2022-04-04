#!/usr/bin/env python3

#author: Siddharth Gollapudi
#email: t-sgollapudi@microsoft.com

import numpy as np

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Any, List, Union

class Distribution(ABC):
    """Interface representing a distribution that the labels of a FilterClass follow."""

    @abstractmethod
    def __init__(self, params: Union[List[Any], int], name: str, size: int):
        self._params = params
        self._name = name
        self._size = size
        self._probabilities = np.arange(0, size)

    @property
    def name(self):
        return self._name

    @property
    def params(self) -> Union[List[Any], int]:
        return self._params

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities

    @probabilities.setter
    def probabilities(self, value: NDArray) -> None:
        self._probabilities = value

    @property
    def size(self) -> int:
        return self._size

    @abstractmethod
    def generate_probabilities(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_samples(self, choices: List[int], num_samples: int) -> NDArray:
        raise NotImplementedError
