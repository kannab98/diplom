#!/usr/bin/env python

from numba import njit, prange
from numba import cuda

import numpy as np
from time import time


# @njit(parallel=True,cache=True)
@cuda.jit
def increment_by_one(an_array):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    if pos < an_array.size:
        an_array[pos] +=1
    return an_array

a = [0 for i in range(100)]
a = increment_by_one(a)
print(a)