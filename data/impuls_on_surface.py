import numpy as np 
import matplotlib.pyplot as plt


H = 3e6
T = 3e-9
c = 3e8
A = 1
x = np.linspace(0,6000)
t0 = np.linspace(0,10*T,100)


plt.ion()
plt.figure()

for i in range(t0.size):
    P = np.zeros(x.size)
    for j in range(x.size):
        offset = H/c*(1 - H/(np.sqrt(x[j]**2 + H**2)))
        if  offset < t0[i] < T+offset:
            P[j] += A
        if offset+T < t0[i] < 2*T+offset:
            P[j] += A
        
    plt.clf()
    plt.ylim((0,2))
    plt.plot(x,P)
    plt.pause(0.5)



plt.show()