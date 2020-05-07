
import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, '..\scripts')
from surface import  Surface
from spectrum import Spectrum


df = pd.read_csv('ku2d.tsv', sep='\t', header=0)
x1 = df.iloc[:,1].values
y1 = df.iloc[:,2].values
h1 = df.iloc[:,3].values
s1 = df.iloc[:,4].values

n = int(x1.size**0.5)

x1 = x1.reshape((n,n))
y1 = y1.reshape((n,n))
h1 = h1.reshape((n,n))
s1 = s1.reshape((n,n))

df = pd.read_csv('c2d.tsv', sep='\t', header=0)
x2 = df.iloc[:,1].values
y2 = df.iloc[:,2].values
h2 = df.iloc[:,3].values
s2 = df.iloc[:,4].values

n = int(x2.size**0.5)

x2 = x2.reshape((n,n))
y2 = y2.reshape((n,n))
h2 = h2.reshape((n,n))
s2 = s2.reshape((n,n))

plt.figure(figsize=(5,4))
plt.contourf(x2,y2,h1,levels=100)
bar = plt.colorbar()
bar.set_label('Высоты C')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plot2dhc.png',dpi=300)




plt.figure(figsize=(5,4))
plt.contourf(x2,y2,h2,levels=100)
bar = plt.colorbar()
bar.set_label('Высоты Ku')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plot2dhku.png',dpi=300)

plt.figure(figsize=(5,4))
plt.contourf(x2,y2,s2,levels=100, vmin = np.min(s1), vmax = np.max(s1))
bar = plt.colorbar()
bar.set_label('Наклоны C')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plot2dsc.png',dpi=300)


plt.figure()
plt.contourf(x1,y1,s1,levels=100)
bar = plt.colorbar()
bar.set_label('Наклоны Ku')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plot2dsku.png',dpi=300)




plt.figure()
plt.contourf(x1,y1,s1-s2,levels=100, vmin = np.min(s1), vmax = np.max(s1))
bar = plt.colorbar()
bar.set_label('Наклоны Ku-C')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plot2dsdelta.png',dpi=300)

plt.show()