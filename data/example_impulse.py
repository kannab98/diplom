import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, '../scripts')
from surface import Surface


file_name =  os.path.basename(sys.argv[0])
(file, ext) = os.path.splitext(file_name)


N = 256
M = 1
t = 0


def plot_surface_2d(x , y, t):
    fig,ax = plt.subplots(nrows = 1, ncols = 1)
    surface = Surface(N=N,M=M,U10=5,wind= np.pi/6)
    x, y = np.meshgrid(x, y)
    z = surface.heights([x,y],t)
    print(z.shape)
    from matplotlib.cm import winter
    plt.contourf(x,y,z,levels=100,cmap=winter)
    plt.colorbar()
    plt.ylabel('Y, м',fontsize=16)
    plt.xlabel('X, м',fontsize=16)
    plt.savefig(file + '.png',  pdi=10**6,transparent=True)

def plot_surface_1d(x , y, t):
    fig,ax = plt.subplots(nrows = 1, ncols = 1)
    surface = Surface(N=N,M=1,U10=10,wind= np.pi/6)
    z = surface.heights([x,0],t)
    print(z.shape)
    plt.plot(x,z)
    plt.ylabel('Z, м',fontsize=16)
    plt.xlabel('X, м',fontsize=16)
    plt.savefig('../fig/' + file + '1d.pdf',  bbox_inches='tight')
    data = pd.DataFrame({"col1":x,"col2":z})
    data.to_csv(file+r'.tsv',index=False,header=None,sep='\t')

x0 = np.linspace(0,2,20000)
y0 = x0
plot_surface_1d(x0,y0,t)
plt.show()
