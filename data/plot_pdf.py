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

N = 1
M = 1

offsetx = 1e9
offsety = 1e10
Xmax  = 500
x0 = np.linspace(-Xmax,Xmax, 128) + offsetx
y0 = np.linspace(-Xmax,Xmax, 1) + offsety
X,Y = np.meshgrid(x0,y0)

wind = 0
data = Data(random_phases = 0, N = N, M=M, band='Ku')
data_surface = data.surface()
data_spectrum = data.spectrum()
surface = Surface(data_surface, data_spectrum)
# h, s, v, cwm = surface.surfaces([X,Y],0)
# cwm_jac = surface.choppy_wave_jac([X,Y],0)

# sigma = np.trapz(surface.spectrum(surface.k0)*surface.k0**2, surface.k0)
# sigma = np.trapz(surface.spectrum(surface.k0), surface.k0)

def parmom(a,b,c):
    S = lambda k: surface.spectrum(k)
    Phi =  lambda k,phi: surface.Phi(k, phi)
    k = surface.k0
    phiint = np.zeros(k.size)
    phi = np.linspace(-np.pi, np.pi, 10**3)
    for i in range(k.size):
        phiint[i] = np.trapz( Phi(k[i],phi)*np.cos(phi)**c*np.sin(phi)**b, phi)
    integral =  np.trapz( np.power(k,a+b-c) * S(k) * phiint, k)
    return integral

def mom(n):
    S = lambda k: surface.spectrum(k)
    k = surface.k0
    integral =  np.trapz( np.power(k,n) * S(k), k)
    return integral
    

sigma = 0.1892384
x = np.linspace(-1,1,100)
W1 = 1/np.sqrt(2*np.pi*sigma)*np.exp(-x**2/sigma)
Sigma = parmom(1,1,1)**2 - parmom(2,0,1)*parmom(0,2,1)
W = W1*(1 + Sigma/mom(0) - mom(1)/mom(0)*x - Sigma/mom(0)**2 * x**2)
plt.plot(x, W)
plt.plot(x, W1)

dictionary = {'zx': x, 'W_lin': W1, 'W_cwm':W}
export = pd.DataFrame(dictionary)
# export.to_csv(os.path.join('data', 'pdf_cwm.csv'), index=False, sep=';')
export.to_csv(os.path.join('pdf_cwm.csv'), index=False, sep=';')

plt.show()
