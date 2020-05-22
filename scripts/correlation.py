import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np
from numpy import pi
from scipy import interpolate,integrate
from tqdm import tqdm
import water



def angles(k,rho):
    k= np.logspace(np.log10(surface.KT[0]), np.log10(surface.KT[-1]), 5*10**4)
    integral=np.zeros(len(rho))
    y=lambda k: k**2*surface.spectrum(k)
    for i in range(len(rho)):
        integral[i]=np.trapz(y(k)*np.cos(k*rho[i]),x=k)
    return integral


def angles_sum(k,rho):
    f=0
    A=surface.amplitude(k)
    k = k[:-1]
    f=np.zeros(len(rho))
    for j in range(len(rho)):
            f[j]=sum( k**2*A**2/2*np.cos(k*rho[j]) )
    return f

def height(k,rho, fourier = 'real'):
    if fourier == 'real':
        S=surface.spectrum(k)
        integral=np.zeros(len(rho))
        for i in range(len(rho)):
            integral[i]=np.trapz(S*np.cos(k*rho[i]),x=k)
        return integral

    if fourier == 'complex':
        S=surface.spectrum(k)
        integral=np.fft.fft(S,n=k.size)
        return integral

def height_sum(k,rho):
    f=0
    f=np.zeros(len(rho))
    A=surface.amplitude(k)
    k = k[:-1]
    for j in range(len(rho)):
            f[j]=sum( A**2/2*np.cos(k*rho[j]) )
    return f

M=1; N=256;
rho = np.linspace(0,200,1000)
surface = water.Surface(N=N, M=M,  KT=[0.05,2000], space='log', whitening='hs')
k = surface.k
H = height_sum(k,rho)
S = angles_sum(k,rho)
surface = water.Surface(N=N, M=M,  KT=[0.05,2000], space='log')
k = surface.k
H1 = height_sum(k,rho)
S1 = angles_sum(k,rho)
klog= np.logspace(np.log10(surface.KT[0]), np.log10(surface.KT[-1]), 10**4)

plt.figure()
plt.plot(rho,height(klog,rho),color='black',label='Теория')
plt.plot(rho,H1, color='blue',label='a')
plt.plot(rho,H, color='red',label='b')
plt.xlabel('$\\rho,~\\text{м}$')
plt.ylabel('$K$')
plt.legend()
plt.savefig('white1.png',dpi=300)
plt.figure()
plt.plot(rho,S1,color='blue',label='a')
plt.plot(rho,S, color='red',label='b')
plt.plot(rho,angles(klog,rho),color='black',label='Теория')
plt.xlabel('$\\rho,~\\text{м}$')
plt.ylabel('$K$')
plt.legend()
plt.savefig('white2.png',dpi=300)
plt.show()

