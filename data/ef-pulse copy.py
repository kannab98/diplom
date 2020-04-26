
import matplotlib.pyplot as plt
import sys
import os    
import numpy as np
from numpy import pi
sys.path.insert(0, '../scripts')
from radiolocator import Pulse
from surface import Surface
import pandas as pd

df = pd.read_csv('heights_big.tsv', sep ='\t', header = None)
z = df.values

df = pd.read_csv('slopesxx_big.tsv', sep ='\t', header = None)
sxx = df.values

df = pd.read_csv('slopesyy_big.tsv', sep ='\t', header = None)
syy = df.values

df = pd.read_csv('x_big.tsv', sep ='\t', header = None)
x  = df.values

df = pd.read_csv('y_big.tsv', sep ='\t', header = None)
y  = df.values

pulse = Pulse()
z0 = 3e6
c = 3e8
omega = 2*pi*20
c = 3e8
k = omega/c

# sxx = sxx*0
# syy = syy*0
# z   = z*0
theta0 = pulse.local_theta_calc(x,y,z,sxx,syy,z0=z0)
x,y,z,sxx,syy = pulse.mirror_sort(x,y,z,sxx,syy,theta0)
theta = pulse.theta_calc(x,y,z,z0=z0)
R = pulse.R(x,y,z,z0=z0)
tau = R/c
tmin = np.min(tau)
timp = 3e-9
t = np.linspace(tmin,tmin+2*timp,2000)
E = np.zeros(t.size)
for i in range(t.size):
    E[i] = pulse.E_calc(t[i],omega,timp,R,theta)
plt.plot(t,E)
plt.show()
