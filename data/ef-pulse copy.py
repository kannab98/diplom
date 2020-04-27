
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

r = np.array([x.flatten(),y.flatten(),z.flatten()])
r0 = np.array([sxx.flatten(),syy.flatten(),-z0*np.ones(x.size)])
n = np.array([sxx.flatten(),syy.flatten(),np.ones(sxx.size)])

R = r - r0

# # print(R.shape)
# nabs = np.sqrt(np.sum(n**2,axis=0))
# Rabs = np.sqrt(np.sum(R**2,axis=0))
# # print(nabs.shape)

# for i in range(nabs.size):
#     theta0 = R[:,i]@n[:,i]
# theta0 *= 1/nabs/Rabs
# theta0 = np.rad2deg(np.arccos(theta0))
# print(theta0)

# for i in range(nabs.size):
#     theta = R[-1,i]
# theta *= 1/Rabs

# theta = theta0[np.where(theta0 < 1)]
# print(theta)
# print(np.tensordot(r-r0,n,axes=0).shape)

theta0 = pulse.local_theta_calc(x,y,z,sxx,syy,z0=z0)
# x,y,z,sxx,syy = pulse.mirror_sort(x,y,z,sxx,syy,theta0)
# theta = pulse.theta_calc(x,y,z,z0=z0)
# R = pulse.R(x,y,z,z0=z0)
# tau = R/c
# tmin = np.min(tau)
# timp = 3e-9
# t = np.linspace(tmin,tmin+7*timp,2000)
# E = np.zeros(t.size)
# for i in range(t.size):
#     E[i] = pulse.E_calc(t[i],omega,timp,R,theta)
# plt.plot(t,E)
# plt.show()
