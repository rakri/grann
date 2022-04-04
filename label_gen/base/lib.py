#!/usr/bin/env python3

#author: Siddharth Gollapudi
#email: t-sgollapudi@microsoft.com

import numpy as np

from typing import List



def _rwh_primes(n: int) -> List[int]:
    '''
    Finds all primes less than n -- credit to RWH

    Uses a basic sieve of Eratosthenes strategy to find the primes.

    If ith entry is still true, set i^2 and all resulting steps by 2*i
    to false.
    '''
    sieve = [True] * n
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i]:
            sieve[i*i::2*i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [i for i in range(3, n, 2) if sieve[i]]


def _approx_nth_prime(n) -> int:
    '''
    Estimates the value of the nth prime (and overshoots, most often)
    '''
    return int((n * np.log(n)) + (n * np.log(np.log(n))))


def first_n_primes(n) -> List[int]:
    num_primes = _approx_nth_prime(n)
    return _rwh_primes(num_primes)[:n]
