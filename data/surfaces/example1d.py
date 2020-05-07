

import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, '..\scripts')
from surface import  Surface
from spectrum import Spectrum


offset = 10e8
x = np.linspace(0+offset,50+offset,100)
y = offset
t = 0
# x,y  = np.meshgrid(x,y)

surface = Surface(1,0,1,1,conf_file='surfaces\surf_ku.ini')
psi = surface.psi
h1   = surface.heights([x,y],t)
sxx1 = surface.slopesxx([x,y],t)
syy1 = surface.slopesyy([x,y],t)
data = pd.DataFrame({'x':x,'heights': h1, 'slopesxx': sxx1, 'slopesyy': syy1})
data.to_csv(r'ku.tsv',sep='\t')

surface = Surface(1,0,1,1,conf_file='surfaces\surf_c.ini' )  
h2   = surface.heights([x,y],t)
sxx2 = surface.slopesxx([x,y],t)
syy2 = surface.slopesyy([x,y],t)
data = pd.DataFrame({'x':x,'heights': h2, 'slopesxx': sxx2, 'slopesyy': syy2})
data.to_csv(r'c.tsv',sep='\t')

