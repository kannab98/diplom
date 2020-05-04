
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os    

from numpy import pi
sys.path.insert(0, '../scripts')
from pulse import Pulse
from surface import Surface


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


# sxx = sxx * 0
# syy = syy * 0
# z = z * 0

r = np.array([x.flatten(),y.flatten(),z.flatten()])
r0 = np.array([np.zeros(x.size),np.zeros(x.size),-z0*np.ones(x.size)])
n = np.array([sxx.flatten(),syy.flatten(),np.ones(sxx.size)])

R = r - r0


theta = pulse.theta_calc(r,r0)
ns = np.sqrt(theta.size).astype(int)
theta0 = pulse.theta0_calc(r,r0,n)

# До сортировки
plt.contourf(r[0,:].reshape(ns,ns),r[1,:].reshape(ns,ns), np.rad2deg(theta0.reshape(ns,ns)) )
bar = plt.colorbar()
bar.set_label('$\\theta_0$, град')
plt.xlabel('X,м')
plt.ylabel('Y,м')



# После сортировки
r,r0,n,theta0,is_mirror = pulse.mirror_sort(r,r0,n,theta0)
print(theta0.size)
                                                                        
                                                                              
plt.scatter(r[0,:],r[1,:],theta0*100,color='r')
# plt.savefig('../fig/model_mirrors4.png',dpi=900,bbox_inches='tight')


R = pulse.R(r-r0).flatten()
tau = R/c
tmin = np.min(tau)
timp = 3e-9
t = np.linspace(tmin,tmin+7*timp,2000)
E = np.zeros(t.size)
for i in range(t.size):
    E[i] = pulse.power(t[i],omega,timp,R,theta)

plt.figure()
plt.plot(t,E/max(E))
plt.xlabel('t, с')
plt.ylabel('P, усл.ед.')
# plt.savefig('../fig/model_impuls4.pdf',bbox_inches='tight')
plt.show()
