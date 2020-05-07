import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, '..\scripts')
from surface import  Surface
from spectrum import Spectrum


offset = 10e8
x = np.linspace(0,25,500)
y = x
t = 0
x,y  = np.meshgrid(x,y)
surface = Surface(1,1,0,0, conf_file='surfaces\surf_ku.ini')
psi = surface.psi
s1 = surface.slopes([x+offset,y+offset],t)
h1 = surface.heights([x+offset,y+offset],t)

data = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'heights': h1.flatten() })
data.to_csv(r'ku_2d_heights.tsv',sep='\t',index=False)
data = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'slopes': s1.flatten() })
data.to_csv(r'ku_2d_slopes.tsv',sep='\t',index=False)

surface = Surface(1,1,0,0, conf_file='surfaces\surf_c.ini' )  
s2 = surface.slopes([x+offset,y+offset],t)
h2 = surface.heights([x+offset,y+offset],t)
data = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'heights': h1.flatten() })
data.to_csv(r'c_2d_heights.tsv',sep='\t',index=False)
data = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(), 'slopes': s1.flatten() })
data.to_csv(r'c_2d_slopes.tsv',sep='\t',index=False)
