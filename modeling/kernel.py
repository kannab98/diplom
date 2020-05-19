from numba import  cuda, float32, float64

import pandas as pd
import numpy as np
from time import time
from time import sleep
from surface import Surface
import math
import matplotlib.pyplot as plt



TPB=16
@cuda.jit
def kernel(ans, x, y, k, phi, A, F, psi):

    i,j = cuda.grid(2)

    if i >= x.size and j >= y.size:
        return

    tmp = 0

    # Выделение памяти в общей памяти блока
    sphi = cuda.shared.array(shape=1, dtype=float32)
    sk  = cuda.shared.array(shape=1, dtype=float32)
    sF = cuda.shared.array(shape=PHI_SIZE, dtype=float32)
    spsi = cuda.shared.array(shape=PHI_SIZE, dtype=float32)

    for n in range(k.size):
        # Загрузка в общую память блока
        sphi[0] = phi[n]
        sk[0] = k[n]
        for m in range(phi.size):
            sF[m] = F[n][m]
            spsi[m] = psi[n][m]
        # Ждем, пока все потоки в блоке закончат загрузку данных
        cuda.syncthreads()
        for m in range(phi.size):
            kr =k[n]*(x[i]*math.cos(phi[m])+y[j]*math.sin(phi[m]))      
            tmp +=  math.cos(kr + spsi[m]) * A[n] * sF[m]
        # Ждем, пока все потоки в блоке закончат вычисления
        cuda.syncthreads()

    ans[j,i] = tmp


        
x   = cuda.to_device(np.linspace(0,100,256) )
y   = cuda.to_device(np.linspace(0,100,256) )



surface = Surface(N=2048, M=256, wind=0, random_phases=0)
k, phi, psi, A_h, A_s, F_h, F_sxx, F_syy = surface.data()
PHI_SIZE = phi.size

k     = cuda.to_device(k)
phi   = cuda.to_device(phi)
psi   = cuda.to_device(psi)
A_h   = cuda.to_device(A_h)
F_h   = cuda.to_device(F_h)




z = np.zeros((x.size,y.size))

threadsperblock = (TPB, TPB)
blockspergrid_x = math.ceil(z.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(z.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
start = time()
kernel[blockspergrid, threadsperblock](z, x, y, k, phi, A_h, F_h, psi)
end = time()
print(end - start)
print(z)
plt.pcolormesh(x,y,z)
plt.show()