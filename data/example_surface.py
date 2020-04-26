import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.insert(0, '../scripts')
from surface import Surface


file_name =  os.path.basename(sys.argv[0])
(file, ext) = os.path.splitext(file_name)


N = 1024
M = 128
t = 0

x = np.linspace(-5000,5000, 100)
y = np.linspace(-5000,5000, 100)
x,y  = np.meshgrid(x,y)
surface = Surface(N=N,M=M,U10=5,wind= 0)

heights = surface.heights([x,y],t)
slopesxx = surface.slopesxx([x,y],t)
slopesyy = surface.slopesyy([x,y],t)


# folder = 'model_data/'
data = pd.DataFrame(heights)
data.to_csv(r'heights_big.tsv',index=False,header=None,sep="\t") 

data = pd.DataFrame(slopesxx)
data.to_csv(r'slopesxx_big.tsv',index=False,header=None,sep="\t") 

data = pd.DataFrame(slopesyy)
data.to_csv(r'slopesyy_big.tsv',index=False,header=None,sep="\t") 

data = pd.DataFrame(x)
data.to_csv(r'x_big.tsv',index=False,header=None,sep="\t") 

data = pd.DataFrame(y)
data.to_csv(r'y_big.tsv',index=False,header=None,sep="\t") 



plt.figure()
plt.contourf(x,y,heights,levels=100,)
plt.figure()
plt.contourf(x,y,slopesxx,levels=100,)
plt.figure()
plt.contourf(x,y,slopesyy,levels=100,)
plt.show()