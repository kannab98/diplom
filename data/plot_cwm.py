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

N = 1024
M = 256

offsetx = 1e9
# offsetx = 0
offsety = 1e10
# offsety = 0
Xmax  = 500
x0 = np.linspace(-Xmax,Xmax, 1000) + offsetx
y0 = np.linspace(-Xmax,Xmax, 1) + offsety
X,Y = np.meshgrid(x0,y0)

wind = 30
data = Data(random_phases = 0, N = N, M=M, band='Ku')
data_surface = data.surface()
data_spectrum = data.spectrum()
surface = Surface(data_surface, data_spectrum)
h,s,v, cwm = surface.surfaces([X,Y],0)
print(h[0].shape, X.shape, cwm[0].shape)
plt.plot(X.T + cwm[0].T, h[0].T)
plt.plot(X.T, h[0].T)

plt.show()
