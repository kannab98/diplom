import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from numpy import pi
from surface import Surface
from data import Data



N = 256
M = 1

offsetx = 1e9
# offsetx = 0
offsety = 1e10
# offsety = 0
Xmax  = 500
x0 = np.linspace(-Xmax,Xmax, 1000) + offsetx
y0 = np.linspace(-Xmax,Xmax, 1) + offsety
X,Y = np.meshgrid(x0,y0)


U = [5, 7, 10, 15]

kfrag = ['lin', 'log', 'quad']
for j in range(len(kfrag)):
    data = {}
    for i in range(len(U)):
        par = Data(random_phases = 0, N = N, M=M, band='Ku', kfrag = kfrag[j], U10=U[i])
        data_surface = par.surface()
        data_spectrum = par.spectrum()
        surface = Surface(data_surface, data_spectrum)
        rho = np.linspace(0,200*(i+1),1000)

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
        k = surface.k
        H = height_sum(k,rho)
        S = angles_sum(k,rho)

        k0 = np.logspace(np.log10(k[0]), np.log10(k[-1]), 10**4)
        H0 = height(k0,rho)
        S0 = height(k0,rho)
        plt.figure()
        plt.plot(rho, H0, color='red',label='a')
        plt.plot(rho, H, color='blue',label='b')
        plt.legend()

        data.update({'rho'+ str(i): rho, 'K_h'+str(U[i]): H/H.max(), 'K0_h'+str(U[i]): H0/H0.max()})
    data = pd.DataFrame(data)
    data.to_csv(os.path.join('data','korr_h_'+kfrag[j]+'.csv'), index = False, sep=';')
plt.show()




