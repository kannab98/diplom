


import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, '..\scripts')
from surface import  Surface
from spectrum import Spectrum


offset = 10e8
x = np.linspace(0,50,500)
y = x
t = 0
x,y  = np.meshgrid(x,y)
surface = Surface(1,1,0,0, conf_file='surfaces\surf_ku.ini')
psi = surface.psi
s1 = surface.slopes([x+offset,y+offset],t)
h1 = surface.heights([x+offset,y+offset],t)
data = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'heights': h1.flatten(), 'slopes': s1.flatten() })
data.to_csv(r'ku2d.tsv',sep='\t')

surface = Surface(1,1,0,0, conf_file='surfaces\surf_c.ini' )  
s2 = surface.slopes([x+offset,y+offset],t)
h2 = surface.heights([x+offset,y+offset],t)
data = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'heights': h2.flatten(), 'slopes': s2.flatten() })
data.to_csv(r'c2d.tsv',sep='\t')

# plt.figure()
# plt.contourf(x,y,s1)
# bar = plt.colorbar()

# plt.figure()
# plt.contourf(x,y,s2)
# bar = plt.colorbar()

# plt.show()