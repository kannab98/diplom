#!/usr/bin/env python

from numba import njit, prange
from numba import cuda, float64, guvectorize, void
import numpy as np
from time import time
import matplotlib.pyplot as plt


# @njit(parallel=True,cache=True)
TPB = 16


@guvectorize([void(float64[:,:], float64[:,:], float64[:,:])], '(m,l),(l,n)->(m,n)', target='cuda')
def dot_gpu(A, B, out):
    """Perform square matrix multiplication of out = A * B
    """

    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        print(tmp,i,j)
        out[i, j] = tmp
 
dot_gpu.max_blocksize = 32

@guvectorize([void(float64[:,:], float64[:,:], float64[:,:])], '(m,l),(l,n)->(m,n)',target='cuda')
def dot_cpu(A, B, out):
    """Perform square matrix multiplication of out = A * B
    """
    m, n = A.shape
    n, p = B.shape
    print(m,n)
    for i in range(m):
        for j in range(p):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            out[i, j] = tmp
dot_cpu.max_blocksize = 32


Mmax = 5
# A=np.random.uniform(size=(Mmax,Mmax))
# B=np.random.uniform(size=(Mmax,Mmax))
A=np.ones((Mmax,Mmax))
B=np.ones((Mmax,Mmax))
out=np.empty((Mmax,Mmax))

# start = time()
# dot_gpu(A,B,out)
# end = time()
# print(end-start)
print(out)

# start = time()
dot_cpu(A,B,out)
# end = time()
# print(end-start)
print(out)

# start = time()
# C = np.dot(A,B)
# end = time()
# print(end-start)
# print(C)

# cpu_time = []
# gpu_time = []
# M = [i for i in range(1000,1000,100)]

# for m in M:

#     A0=np.random.uniform(-100,100,size=(m,m))
#     B0=np.random.uniform(-100,100,size=(m,m))
#     C0=np.empty((m,m))

#     start = time()
#     dot(A0,B0,C0)
#     end = time()
#     gpu_time.append(end-start)

#     start = time()
#     np.dot(A0,B0)
#     end = time()
#     cpu_time.append(end-start)


# plt.plot(M,cpu_time)
# plt.plot(M,gpu_time)
# plt.show()