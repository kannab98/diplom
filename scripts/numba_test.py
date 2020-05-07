from numba import njit, prange
from numba import cuda

import numpy as np
from time import time
from surface import Surface


surface = Surface(1,0,0,0, N=4, M=4)
@njit(parallel=True,cache=True)
# @cuda.jit
def test(A,x,y,phi,psi):
    acc = np.empty((x.size,y.size))
    for i in prange(x.size):
        for j in prange(y.size):
            for n in prange(A.shape[0]):
                for m in prange(A.shape[1]):
                    # kr = k[n]*(x[i]*np.cos(phi[m])+y[j]*np.sin(phi[m]))
                    kr = 0
                    acc[i,j] += A[n,m] * np.cos(kr + psi[n,m])
    return acc

N = surface.N
M = surface.N
A = surface.A
psi = surface.psi
phi = surface.phi
F = surface.F
x   = np.linspace(0,1,100)
y   = np.linspace(0,1,100)


start = time()
test(F,x,y,phi,psi)
stop = time()
print(stop - start)
