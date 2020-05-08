import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('gpu_cpu_calc.tsv', sep='\t', header=0)
x = df.iloc[:,0].values
gpu = df.iloc[:,1].values
cpu = df.iloc[:,2].values

plt.plot(x,gpu, label='GPU')
plt.plot(x,cpu, label='CPU')

plt.xlabel('Размер координатной сетки ')
plt.ylabel('Время')
plt.legend()

plt.figure()
plt.title('Относительная скорость вычислений')
plt.plot(x,cpu/gpu)
plt.xlabel('Размер координатной сетки ')
plt.ylabel('cpu/gpu')
plt.show()