from numba import njit, prange, cuda, float32,float64, void,guvectorize

import numpy as np
from time import time
from time import sleep
from surface import Surface
import math
import matplotlib.pyplot as plt



@njit(parallel=True,cache=False)
def kernel_cpu(acc,x,y,phi,psi,k,A,F):
    for i in prange(x.size):
        for j in prange(y.size):
            for n in prange(F.shape[0]):
                for m in prange(F.shape[1]):
                    kr = k[n]*(x[i]*np.cos(phi[m])+y[j]*np.sin(phi[m]))
                    acc[i,j] += F[n,m] * np.cos(kr + psi[n,m]) * A[n]
    return acc

# @cuda.jit
# def kernel_gpu(acc,x,y,phi,psi,k,A,F):


#     i,j = cuda.grid(2)

#     if i < acc.shape[0] and j < acc.shape[1]:
#         for n in range(k.size):
#             for m in range(phi.size):
#                 kr = k[n]*(x[i]*math.cos(phi[m])+y[j]*math.sin(phi[m]))
#                 acc[i,j] += F[n,m] * math.cos(kr + psi[n,m]) * A[n]
TPB = 16
def kernel_gpu(acc,x,y,k):

    i,j = cuda.grid(2)

    if i >= x.size and j >= y.size:
        return

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    tmp = 0.
    for n in range(k.size):
        tmp += k[n]*(x[i]+y[j])
    



surface = Surface(1,0,0,0, N=16, M=16, wind=0,random_phases=0)
N = surface.N
M = surface.N
A = surface.A
psi = surface.psi
phi = surface.phi
F = surface.F
offset = 1e8
x   = np.linspace(0,100,16) + offset
y   = np.linspace(0,100,16) + offset
k = surface.k


an_array = np.zeros((x.size,y.size))
threadsperblock = (TPB, TPB)
blockspergrid_x = math.ceil(k.size / threadsperblock[0])
blockspergrid_y = math.ceil(phi.size / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
start = time()
# kernel_gpu[blockspergrid, threadsperblock](an_array,x,y,phi,psi,k,A,F)
kernel_gpu[blockspergrid, threadsperblock](an_array,x,y)
stop = time()
print(stop - start)
plt.figure()
X,Y = np.meshgrid(x,y)
# plt.contourf(X,Y,an_array,levels=100)
plt.pcolormesh(X,Y,an_array)
plt.colorbar()



# plt.figure()
# start = time()
# an_array=kernel_cpu(an_array,x,y,phi,psi,k,A,F)
# stop = time()
# print(stop - start)
# # plt.contourf(X,Y,an_array,levels=100)
# plt.pcolormesh(X,Y,an_array)
# plt.colorbar()

# plt.figure()
# a=surface.heights([X,Y],0)
# plt.pcolormesh(X,Y,a)
# plt.colorbar()
plt.show()



