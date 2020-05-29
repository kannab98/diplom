import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from numpy import pi
from spectrum import Spectrum



band = ['Ku' , 'C']
U  = [5, 10, 15]
for i in range(len(band)):
    data = {}
    plt.figure()
    for j in range(len(U)):
        spectrum = Spectrum([U[j], 20170, band[i], None])
        S = spectrum.get_spectrum()
        k = spectrum.k0[0:-1:100]
        plt.loglog(k, S(k))



        data.update({'k'+str(U[j]): k })
        data.update({'S'+str(U[j]): S(k) })
    data = pd.DataFrame(data)
    data.to_csv(os.path.join('data','spectrum_'+band[i]+'.csv'), index = False, sep=';')

plt.show()

