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

data = Data(wind=0, N=1, M=2048, U10=10)
surf = data.surface()
spec = data.spectrum()
surface = Surface(surf, spec)

phi = surface.phi_c
k_m = surface.k_m
k = [0.5*k_m, k_m, 2*k_m, 10*k_m, 50*k_m, 100*k_m]
F_key = ['Fkm/2', 'Fkm', 'F2km', 'F10km', 'F50km', 'F100km']
dictionary = {'phi': phi}

for i in range(len(k)):
    F = surface.Phi(k[i], phi)
    dictionary.update({F_key[i]: F})

export = pd.DataFrame(dictionary)
export.to_csv(os.path.join('data', 'angles_distrib.csv'), index=False, sep=';')






# data = {k_key[i]: k[i] for i in range(len(k))}
# data.update({S_key[i]: S[i] for i in range(len(S))})
# data = pd.DataFrame(data)
# data.to_csv(os.path.join('data','spectrum_c.csv'), index = False, sep=';')
# plt.show()


