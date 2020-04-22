import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from surface import Surface
sys.path.insert(0, '../scripts')


file_name =  os.path.basename(sys.argv[0])
(file, ext) = os.path.splitext(file_name)


N = 256
M = 1
t = 0


def plot_surface(x , y, t):
    fig,ax = plt.subplots(nrows = 1, ncols = 1)
    surface = water.Surface(N=N,M=M,U10=5,wind= np.pi/6)
    x, y = np.meshgrid(x, y)
    z = surface.model([x,y],t)
    print(z.shape)
    from matplotlib.cm import winter
    plt.contourf(x,y,z,levels=100,cmap=winter)
    plt.colorbar()
    plt.ylabel('Y, м',fontsize=16)
    plt.xlabel('X, м',fontsize=16)
    plt.savefig('/home/kannab/documents/water/poster/fig/water5.png',  pdi=10**6,transparent=True)

x0 = np.linspace(0,200,200)
y0 = x0
plot_surface(x0,y0,t)
plt.show()
