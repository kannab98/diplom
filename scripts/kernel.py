from numba import  cuda, float32, float64
import os
import pandas as pd
import numpy as np
from time import time
import datetime
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
    # sphi = cuda.shared.array(shape=1, dtype=float32)
    # sk  = cuda.shared.array(shape=1, dtype=float32)
    # sF = cuda.shared.array(shape=PHI_SIZE, dtype=float32)
    # spsi = cuda.shared.array(shape=PHI_SIZE, dtype=float32)

    for n in range(k.size):
        # Загрузка в общую память блока
        # sphi[0] = phi[n]
        # sk[0] = k[n]
        # for m in range(phi.size):
        #     sF[m] = F[n][m]
        #     spsi[m] = psi[n][m]
        # Ждем, пока все потоки в блоке закончат загрузку данных
        cuda.syncthreads()
        for m in range(phi.size):
            kr = k[n]*(x[i]*math.cos(phi[m])+y[j]*math.sin(phi[m]))      
            tmp +=  math.cos(kr + psi[n][m]) * A[n] * F[n][m]
        # Ждем, пока все потоки в блоке закончат вычисления
        cuda.syncthreads()

    ans[j,i] = tmp



@cuda.jit
def kernelxx(ans, x, y, k, phi, A, F, psi):

    i,j = cuda.grid(2)

    if i >= x.size and j >= y.size:
        return

    tmp = 0

    for n in range(k.size):
        cuda.syncthreads()
        for m in range(phi.size):
            kr = k[n]*(x[i]*math.cos(phi[m]))      
            tmp +=  math.cos(kr + psi[n][m]) * A[n] * F[n][m]
        cuda.syncthreads()

    ans[j,i] = tmp


@cuda.jit
def kernelyy(ans, x, y, k, phi, A, F, psi):

    i,j = cuda.grid(2)

    if i >= x.size and j >= y.size:
        return

    tmp = 0

    for n in range(k.size):
        cuda.syncthreads()
        for m in range(phi.size):
            kr = k[n]*(y[i]*math.sin(phi[m]))      
            tmp +=  math.cos(kr + psi[n][m]) * A[n] * F[n][m]
        cuda.syncthreads()

    ans[j,i] = tmp


        
surface = Surface(1,1,1,1, N=2048, M=256, wind=30, random_phases=0, band = 'C')

k = surface.k
phi = surface.phi
A = surface.A
A_s = surface.A_slopes
F_s = surface.F_slopes
A_sxx = surface.A_slopesxx
F_sxx = surface.F_slopes

A_syy = surface.A_slopesxx
F_syy = surface.F_slopes

F = surface.F
PHI_SIZE = F.shape[1]
psi = surface.psi

k = cuda.to_device(k)
phi = cuda.to_device(phi)

A = cuda.to_device(A)
A_s = cuda.to_device(A_s)
A_sxx = cuda.to_device(A_sxx)
A_syy = cuda.to_device(A_syy)

F = cuda.to_device(F)
F_s = cuda.to_device(F_s)
F_sxx = cuda.to_device(F_sxx)
F_syy = cuda.to_device(F_syy)

psi = cuda.to_device(psi)


offsetx = 1e9*np.random.uniform()
offsety = 1e9*np.random.uniform()
x0 = np.linspace(0,50,128) + offsetx
y0 = np.linspace(0,50,128) + offsety
x = cuda.to_device(x0)
y = cuda.to_device(y0)


threadsperblock = (TPB, TPB)
blockspergrid_x = math.ceil(x.size / threadsperblock[0])
blockspergrid_y = math.ceil(y.size / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)


print('Поверхность высот')
h = np.zeros((x.size,y.size))
kernel[blockspergrid, threadsperblock](h,x,y,k,phi,A,F,psi)

print('Поверхность наклонов')
s = np.zeros((x.size,y.size))
kernel[blockspergrid, threadsperblock](s,x,y,k,phi,A_s,F_s,psi)

print('Поверхность наклонов X')
sxx = np.zeros((x.size,y.size))
kernel[blockspergrid, threadsperblock](sxx,x,y,k,phi,A_sxx,F_sxx,psi)

print('Поверхность наклонов Y')
syy = np.zeros((x.size,y.size))
kernel[blockspergrid, threadsperblock](syy,x,y,k,phi,A_syy,F_syy,psi)

x = x0 - offsetx
y = y0 - offsety

# print(np.std(z),np.mean(z))

# plt.figure()
# X,Y = np.meshgrid(x,y)
# start = time()
# Z = surface.heights([X,Y],0) 
# end = time()



datadir = 'data' + datetime.datetime.now().strftime("%m%d_%H%M")
os.makedirs(datadir)

modeldatadir = os.path.join(datadir,'model_data')
premodeldatadir = os.path.join(datadir,'parameters')
os.makedirs(modeldatadir)
os.makedirs(premodeldatadir)

#############################################
x,y = np.meshgrid(x,y)
data = pd.DataFrame({'x': x.flatten(),'y': y.flatten(), 'heights': h.flatten() })
data.to_csv(os.path.join(modeldatadir,'heights.tsv'), index=False, sep='\t')

data = pd.DataFrame({'x': x.flatten(),'y': y.flatten(), 'slopes': s.flatten() })
data.to_csv(os.path.join(modeldatadir,'slopes.tsv'), index=False, sep='\t')

data = pd.DataFrame({'x': x.flatten(),'y': y.flatten(), 'slopesxx': sxx.flatten() })
data.to_csv(os.path.join(modeldatadir,'slopesxx.tsv'), index=False, sep='\t')

data = pd.DataFrame({'x': x.flatten(),'y': y.flatten(), 'slopesxx': syy.flatten() })
data.to_csv(os.path.join(modeldatadir,'slopesyy.tsv'), index=False, sep='\t')

plt.figure()
plt.contourf(x,y,h,levels=100)
plt.xlabel('x')
plt.ylabel('y')
bar = plt.colorbar()
bar.set_label('высоты')
plt.savefig(os.path.join(modeldatadir,'heights.png'), dpi=300, bbox_inches='tight')

plt.figure()
plt.contourf(x,y,s,levels=100)
plt.xlabel('x')
plt.ylabel('y')
bar = plt.colorbar()
# bar.set_label('наклоны')
plt.savefig(os.path.join(modeldatadir,'slopes.png'), dpi=300, bbox_inches='tight')

plt.figure()
plt.contourf(x,y,sxx,levels=100)
plt.xlabel('x')
plt.ylabel('y')
bar = plt.colorbar()
bar.set_label('наклоны X')
plt.savefig(os.path.join(modeldatadir,'slopesxx.png'), dpi=300, bbox_inches='tight')


plt.figure()
plt.contourf(x,y,syy,levels=100)
plt.xlabel('x')
plt.ylabel('y')
bar = plt.colorbar()
bar.set_label('наклоны Y')
plt.savefig(os.path.join(modeldatadir,'slopesyy.png'), dpi=300, bbox_inches='tight')

plt.figure()
plt.loglog(surface.k, surface.spectrum(k))
plt.xlabel('k')
plt.ylabel('S')
plt.savefig(os.path.join(modeldatadir,'spectrum.png'), dpi=300, bbox_inches='tight')

#############################################
data = pd.DataFrame({'k': surface.k})
data.to_csv(os.path.join(premodeldatadir,'k.tsv'), index=True, sep='\t')
data = pd.DataFrame({'A': surface.A})
data.to_csv(os.path.join(premodeldatadir,'A.tsv'), index=True, sep='\t')
data = pd.DataFrame({'phi': surface.phi})
data.to_csv(os.path.join(premodeldatadir,'phi.tsv'), index=True, sep='\t')
data = pd.DataFrame({'psi': surface.psi.flatten()})
data.to_csv(os.path.join(premodeldatadir,'psi.tsv'), index=True, sep='\t')
data = pd.DataFrame({'F': surface.F.flatten()})
data.to_csv(os.path.join(premodeldatadir,'F.tsv'), index=True, sep='\t')

sigma = np.sum(surface.A**2/2)
data  = pd.DataFrame({'Band': surface.band,
                      'N': surface.N, 
                      'M':surface.M, 
                      'U':surface.U10, 
                      'sigma^2-practice': sigma,
                      'sigma^2-theory': surface.sigma_sqr},index=[0],)
data.to_csv(os.path.join(premodeldatadir,'params'+'.tsv'), index=False, sep='\t')

# plt.contourf(X,Y,Z,levels=100)
# print(std2/std1)

plt.show()