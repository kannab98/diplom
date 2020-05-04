from numba import njit, prange
from numba import cuda

import numpy as np
from time import time


# @njit(parallel=True,cache=True)
@cuda.jit
def test(A,x,y,phi,psi):
    acc = np.empty((x.size,y.size))
    for i in prange(x.size):
        for j in prange(y.size):
            for n in prange(A.shape[0]):
                for m in prange(A.shape[1]):
                    kr = k[n]*(x[i]*np.cos(phi[m])+y[j]*np.sin(phi[m]))
                    acc[i,j] += A[n,m] * np.cos(kr + psi[n,m])
    return acc

N = 2048
M = 128
A = np.random.uniform(size=(N,M))
psi = np.random.uniform(-np.pi,np.pi,size=(N,M))
phi = np.linspace(-np.pi,np.pi,M)
k = np.random.uniform(size=N)
x   = np.linspace(0,1,100)
y   = np.linspace(0,1,100)
y   = np.linspace(0,1,100)


start = time()
test(A,x,y,phi,psi)
stop = time()
print(stop - start)
