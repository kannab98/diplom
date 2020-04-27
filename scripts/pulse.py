import numpy as np
class Pulse:
    def __init__(self, r0 = [0,0,3e6],
                       gane_width=np.deg2rad(1.5),
                       c = 3e8):
        self.c =  c
        self.r0 = r0
        #!$gane\_width \equiv \theta_{3dB}$!
        gane_width = np.deg2rad(1.5) # Ширина диаграммы направленности в радианах
        self.gamma = 2*np.sin(gane_width/2)**2/np.log(2)

    def G(self,theta,G0=1):
            # G -- диаграмма направленности
            # theta -- угол падения
            return G0*np.exp(-2/self.gamma * np.sin(theta)**2)
    
    def R(self, R):
        Rabs = np.sqrt(np.sum(R**2,axis=0))
        return Rabs 

    def N(self,n):
        N = np.sqrt(np.sum(n**2,axis=0))
        return N

    def theta_calc(self, r, r0):
        R = r - r0
        theta = R[-1,:]/self.R(R)
        return np.arccos(theta)

    def theta0_calc(self,r,r0,n):
        R = r - r0
        Rabs = self.R(R)
        theta0 = np.einsum('ij, ij -> j', R, n)
        theta0 *= 1/Rabs/self.N(n)
        return np.arccos(theta0)

    def mirror_sort(self,r,r0,n,theta, err = 1):

        index = np.where(theta < np.deg2rad(2))
        r     = r [:, index]
        r0    = r0[:, index]
        n     = n [:, index]
        theta = theta[index]

        return r, r0, n, theta

    def power(self, t, omega ,timp ,R, theta,):
        
        c = self.c 
        G = self.G
        tau = R/c

        index = [ i for i in range(tau.size) if 0 <= t - tau[i] <= timp ]
        theta = theta[index] 
        R = R[index]
        tau = tau[index]
        # Путь к поверхности
        #! $\omega\tau\cos(\theta) = kR$! 
        E0 = G(theta)/R
        e0 = np.exp(1j*omega*(t  - tau*np.cos(theta)) ) 
        # Путь от поверхности
        E0 = E0*G(theta)/R
        e0 = np.exp(1j*omega*(tau+ tau*np.cos(theta)) ) 
        
        
        return np.sum(E0*e0)**2/2
    



