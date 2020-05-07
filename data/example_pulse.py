import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os    

from numpy import pi
from numpy import random
sys.path.insert(0, '../scripts')
from pulse import Pulse
from surface import Surface


N = 1024
M = 128
t = 0

surface = Surface(1,0,1,1, N=N, M=M, conf_file='surfaces/surf_ku.ini')



pulse = Pulse()
z0 = 3e6
c = 3e8
omega = 2*pi*20
k = omega/c
timp = 3e-9
T0 = z0/c
Xmax = 5000
T = np.linspace(T0-timp,np.sqrt(z0**2+Xmax**2)/c,200)
P = np.zeros(T.size)



mirrors = 0

while mirrors<10:
    x = random.uniform(-Xmax, Xmax, size=(100))
    y = random.uniform(-Xmax, Xmax, size=(100))
    z = surface.heights([x,y],t)
    zxx = surface.slopesxx([x,y],t)
    zyy = surface.slopesyy([x,y],t)

    r = np.array([x.flatten(),y.flatten(),z.flatten()])
    r0 = np.array([np.zeros(x.size),np.zeros(x.size),-z0*np.ones(x.size)])
    n = np.array([zxx.flatten(),zyy.flatten(),np.ones(zxx.size)])

    R = r - r0
    theta = pulse.theta_calc(r,r0)
    theta0 = pulse.theta0_calc(r,r0,n)
    r,r0,n,theta0,is_mirror = pulse.mirror_sort(r,r0,n,theta0)

    if is_mirror != 0:
        mirrors += is_mirror
        print(mirrors)
        R = pulse.R(r-r0).flatten()
        tau = np.array(R/c)
        tmin = np.min(tau)
        for i in range(T.size):
            P[i] += pulse.power(T[i],omega,timp,R,theta)

        plt.ion()
        plt.clf()
        plt.plot(T,P)
        plt.show()
        plt.pause(20)

plt.savefig('example-impulse.png')
data = pd.DataFrame({'P': P, 'T': T})
data.to_csv(r'impulse.tsv',index=False, sep="\t") 

