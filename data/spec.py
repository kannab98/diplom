
import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.insert(0, '../scripts')
from spectrum import Spectrum 

spectrum = Spectrum('surfaces/surf_ku.ini', band = 'Ku')
k = spectrum.k0
S = spectrum.get_spectrum()
plt.loglog(k,S(k),label='kU')

spectrum = Spectrum('surfaces/surf_ku.ini', band='X')
k = spectrum.k0
S = spectrum.get_spectrum()
plt.loglog(k,S(k),label='X')

spectrum = Spectrum('surfaces/surf_ku.ini', band='C')
k = spectrum.k0
S = spectrum.get_spectrum()
plt.loglog(k,S(k),label='C')

plt.show()

