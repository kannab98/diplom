import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, '..\scripts')
from surface import  Surface
from spectrum import Spectrum


df = pd.read_csv('ku.tsv', sep='\t', header=0)
x1 = df.iloc[:,1].values
h1 = df.iloc[:,2].values
s1 = df.iloc[:,3].values


df = pd.read_csv('c.tsv', sep='\t', header=0)
x2 = df.iloc[:,1].values
h2 = df.iloc[:,2].values
s2 = df.iloc[:,3].values



plt.figure(figsize=(5,4))
plt.plot(x1,h1, label='Ku')
plt.plot(x1,h2, label='C')
plt.xlabel('$X,$ м')
plt.ylabel('$Высоты,$ м')
plt.legend()
plt.savefig('plot1dh.png',dpi=300)

plt.figure(figsize=(5,4))
plt.plot(x1,s1,label='Ku')
plt.plot(x1,s2,label='C')
plt.xlabel('$X,$ м')
plt.ylabel('$Наклоны$')
plt.legend()
plt.savefig('plot1ds.png',dpi=300)


plt.show()