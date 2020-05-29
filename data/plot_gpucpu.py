import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv('gpucalc.tsv', sep='\t', header=0)
N = df.iloc[:,0].values
t1 = df.iloc[:,1].values
t2 = df.iloc[:,2].values

plt.plot(N,t1)
plt.plot(N,t2)
plt.show()