
import numpy as np
import math
from numba import  cuda
# Мои классы
from surface import Surface
from data import Data




TPB=16
# Ядро устройства для быстрых параллельных вычислений на GPU
# ans [0] -- поверхность высот 
# ans [1] -- поверхность уклонов
# ans [2] -- поверхность уклонов по X
# ans [3] -- поверхность уклонов по Y
@cuda.jit
def kernel(ans, x, y, k, phi, A, F, psi):

    i,j = cuda.grid(2)

    if i >= x.size and j >= y.size:
        return

    for n in range(k.size):
        for m in range(phi.size):
            kr = k[n]*(x[i]*math.cos(phi[m]) + y[j]*math.sin(phi[m]))      
            tmp =  math.cos(kr + psi[n][m]) * A[n] * F[n][m]
            tmp1 = - math.sin(kr + psi[n][m]) * A[n] * F[n][m]
            ans[0,j,i] +=  tmp
            ans[1,j,i] +=  tmp1 * k[n]
            ans[2,j,i] +=  tmp1 * k[n] * math.cos(phi[m])
            ans[3,j,i] +=  tmp1 * k[n] * math.sin(phi[m])



N = 256
M = 128
wind = 30

x0 = np.linspace(-Xmax,Xmax, 1024) 
y0 = np.linspace(-Xmax,Xmax, 1024) 
x = cuda.to_device(x0)
y = cuda.to_device(y0)


# Параметры модели
data = Data(random_phases = 0, N = N, M=M, band='Ku',wind=wind, U10=5)
data_surface = data.surface()
data_spectrum = data.spectrum()
# Вычисление распределения по углу, по частоте и счет амплитуд гармоник
surface = Surface(data_surface, data_spectrum)

k = surface.k 
phi = surface.phi
A = surface.A
F = surface.F
PHI_SIZE = F.shape[1]
psi = surface.psi

# Создание массива типа gpuarray
k = cuda.to_device(k)
phi = cuda.to_device(phi)
A = cuda.to_device(A)
F = cuda.to_device(F)
psi = cuda.to_device(psi)


# Количество потоков на блок
threadsperblock = (TPB, TPB)
# Вычисление необходимого количества блоков и создание двумерной сетки 
blockspergrid_x = math.ceil(x.size / threadsperblock[0])
blockspergrid_y = math.ceil(y.size / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)


surfaces = np.zeros((4, x.size, y.size))

# Вызов ядра на видеокарте
kernel[blockspergrid, threadsperblock](surfaces, x, y, k, phi, A, F, psi)
