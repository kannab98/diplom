
import matplotlib.pyplot as plt
import sys
import os    
import numpy as np
from numpy import pi
sys.path.insert(0, '../scripts')
from radiolocator import Pulse
from surface import Surface
from matplotlib.cm import winter,summer,Greys

pulse = Pulse()
# theta = np.linspace(-pi/6,pi/6,1000)
# G = pulse.G(theta=theta)
# plt.plot(
#     np.deg2rad(theta),
#     G
# )

x = np.linspace(-10000,10000,100)
y = np.linspace(-10000,10000,100)
X,Y = np.meshgrid(x,y)
H = 3e6

def A(t,tau,timp=3e-8):
    if 0 <= t-tau <= timp:
        A = 1
    else:
        A = 0
    return A

def E(t,x,y,z,timp=3e-9):

    omega = 2*pi*20
    c = 3e8
    k = omega/c
    theta = pulse.theta_calc(x,y,z)
    R = pulse.R(x,y,z)
    tau = R/c
    E = np.zeros(R.shape)
    Es = 0
    G = pulse.G
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if t - tau[i,j] < 0:
                E[i,j] = 0
            else:
                E[i,j] =  A(t,tau[i,j],timp)*G(theta[i,j])/R[i,j] * \
                    np.exp(
                        1j*omega*t 
                        - omega/c*R[i,j]*np.cos(theta[i,j])
                    )
            # if theta[i,j] < np.deg2rad(1):
            Es += E[i,j]/R[i,j]  *G(theta[i,j])* \
                np.exp(
                    1j*omega*(tau[i,j]) 
                    - 1j*omega/c*R[i,j]*np.cos(theta[i,j])
                )

    return Es

timp = 3e-9
t = pulse.R(X,Y,H)/c
tmin = np.min(t)
tmax = np.max(t)
print(tmin,tmax)
T = np.linspace(tmin,tmin+3*timp,128)
# T = [tmin]




plt.figure()
Ee = []
for t in T:
    plt.ion()
    plt.clf()
    e = E(t,X,Y,H,timp)
    # Ee.append(e)
    plt.contourf(X,Y,e.real,levels=100)
    # plt.colorbar()
    # plt.pause(0.5)




plt.plot(T,Ee)
plt.show()