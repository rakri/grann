#!/usr/bin/env python3

#author: Siddharth Gollapudi
#email: t-sgollapudi@microsoft.com

import numpy as np

from base.distribution import Distribution
from numpy.typing import NDArray
from scipy.stats import expon, lognorm, norm
from typing import List, Any



class LogNormal(Distribution):
    """ Class representing a lognormal distribution, with instance specific parameters. """
    def __init__(self, params: List[Any], name: str, size: int):
        super().__init__(params, name, size)
        self.generate_probabilities()


    def generate_probabilities(self) -> None:
        '''
        Generates a list of probabilities, fitting the size, parameters, and type
        of distribution.

        Note that this method of "discretizing" may be slightly inaccurate for small
        sizes.
        '''
        upper = np.add(self.probabilities, 0.5)
        lower = np.add(self.probabilities, -0.5)
        filter_probs = lognorm.cdf(upper, *self.params) - lognorm.cdf(lower, *self.params)
        self.probabilities = np.divide(filter_probs, sum(filter_probs))


    def draw_samples(self, choices: List[int], num_samples: int) -> NDArray:
        '''
        Given the probabilities generated, and a number of choices equaling the size of
        the distribution, perform num_samples samples without replacement.
        '''
        if len(choices) != len(self.probabilities):
            raise ValueError("Length of choices must equal distribution size")
        return np.random.choice(choices, num_samples, replace=False, p=self.probabilities)



class Normal(Distribution):
    """ Class representing a normal distribution, with instance specific parameters. """
    def __init__(self, params: List[Any], name: str, size: int):
        super().__init__(params, name, size)
        self.generate_probabilities()


    def generate_probabilities(self) -> None:
        '''
        Check LogNormal documentation
        '''
        upper = np.add(self.probabilities, 0.5)
        lower = np.add(self.probabilities, -0.5)
        filter_probs = norm.cdf(upper, *self.params) - norm.cdf(lower, *self.params)
        self.probabilities = np.divide(filter_probs, sum(filter_probs))


    def draw_samples(self, choices: List[int], num_samples: int) -> NDArray:
        '''
        Check LogNormal documentation
        '''
        if len(choices) != len(self.probabilities):
            raise ValueError("Length of choices must equal distribution size")
        return np.random.choice(choices, num_samples, replace=False, p=self.probabilities)



class Uniform(Distribution):
    """ Class representing a uniform distribution, with instance specific parameters. """
    def __init__(self, params: List[Any], name: str, size: int):
        super().__init__(params, name, size)
        self.generate_probabilities()


    def generate_probabilities(self) -> None:
        '''
        Check LogNormal documentation
        '''
        filter_probs = np.ones(self.size)
        self.probabilities = np.divide(filter_probs, sum(filter_probs))


    def draw_samples(self, choices: List[int], num_samples: int) -> NDArray:
        '''
        Check LogNormal documentation
        '''
        if len(choices) != len(self.probabilities):
            raise ValueError("Length of choices must equal distribution size")
        return np.random.choice(choices, num_samples, replace=False, p=self.probabilities)



class Exponential(Distribution):
    """ Class representing a exponential distribution, with instance specific parameters. """
    def __init__(self, params: List[Any], name: str, size: int):
        super().__init__(params, name, size)
        self.generate_probabilities()


    def generate_probabilities(self) -> None:
        '''
        Check LogNormal documentation
        '''
        upper = np.add(self.probabilities, 0.5)
        lower = np.add(self.probabilities, -0.5)
        filter_probs = expon.cdf(upper, *self.params) - expon.cdf(lower, *self.params)
        self.probabilities = np.divide(filter_probs, sum(filter_probs))


    def draw_samples(self, choices: List[int], num_samples: int) -> NDArray:
        '''
        Check LogNormal documentation
        '''
        if len(choices) != len(self.probabilities):
            print(self.probabilities)
            raise ValueError("Length of choices must equal distribution size")
        return np.random.choice(choices, num_samples, replace=False, p=self.probabilities)



class Constant(Distribution):
    """ Just returns the same number every time."""
    def __init__(self, params: int, name: str):
        super().__init__(params, name, 0)


    def generate_probabilities(self) -> None:
        pass


    def draw_samples(self, choices: List[int], num_samples: int) -> NDArray:
        '''
        Special case: merely returns a list of the same number, with num_samples
        elements.
        '''
        return np.array([self.params] * num_samples)
