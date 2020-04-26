import matplotlib.pyplot as plt
import sys
import os    
import numpy as np
from numpy import pi
sys.path.insert(0, '../scripts')
from surface import Surface
import pandas as pd
from matplotlib.cm import winter,summer,Greys

surface = Surface(N=2048,M=128,U10=5)
x0 = np.linspace(0,200,200)
y0 = x0
t=0
x, y = np.meshgrid(x0, y0)


df = pd.read_csv('heights.tsv', sep ='\t', header = None)
heights = df.values

df = pd.read_csv('slopesxx.tsv', sep ='\t', header = None)
slopesxx = df.values

df = pd.read_csv('slopesyy.tsv', sep ='\t', header = None)
slopesyy = df.values

df = pd.read_csv('x.tsv', sep ='\t', header = None)
x  = df.values

df = pd.read_csv('y.tsv', sep ='\t', header = None)
y  = df.values

h = 1e6
offset = 100
theta = (z1*(x-offset) + z2*(y-offset) - z3 + h)/np.sqrt((x-offset)**2+(y-offset)**2+(z3-h)**2)/np.sqrt(z1**2+z2**2+1)
theta = np.rad2deg(np.arccos(theta))
mirror = np.zeros(theta.shape)
for i in range(theta.shape[0]):
    for j in range(theta.shape[1]):
        if  theta[i,j] < 1:
            mirror[i,j] = 1


plt.figure(1,figsize=(7,6))
plt.contourf(x,y,z3,levels=100,cmap=winter)
bar = plt.colorbar()
bar.set_label('$Z,$ м')
plt.xlabel('$X,$ м')
plt.ylabel('$Y,$ м')

plt.figure(2,figsize=(7,6))
plt.contourf(x,y,theta,levels=100,cmap=Greys)
bar = plt.colorbar()
bar.set_label('$\\theta,$ градусы')
plt.xlabel('$X,$ м')
plt.ylabel('$Y,$ м')

plt.figure(3,figsize=(6,6))
plt.scatter(x,y,20*mirror,c = 'r',marker='o')
# plt.legend('квазизеркальная точка')
plt.xlabel('$X,$ м')
plt.ylabel('$Y,$ м')


plt.show()
