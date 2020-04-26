import numpy as np
from numpy import pi
from scipy.optimize import curve_fit
from scipy.special import erf

# import pandas.DataFrame


xrad = yrad = 0

class Radiolocator():
    def __init__(self, h=1e6, xi=0.0, theta=1.5, c=299792458, sigma=1, 
                    angles_in='degrees', pulse = np.array([None]), t = None):
        self.R = 6370e3 # Радиус Земли в метрах
        self.c = c # Скорость света в м/с
        if angles_in=='degrees':
            self.xi = np.deg2rad(xi) # Отклонение антенны в радианах
            self.theta = np.deg2rad(theta) # Ширина диаграммы направленности в радианах
            self.Gamma = self.gamma(self.theta)

        if pulse.any() != None:
            params = radiolocator.calc(t,pulse)
            print('xi = {},\nh = {},\nsigma_s = {}'.format(params[0],params[1],params[2]))

        else:
            self.h = h # Высота орбиты в метрах
            self.sigma_s = sigma # Дисперсия наклонов


        self.T = 3e-9 # Временное разрешение локатора


    def H(self,h):
        return h*( 1+ h/self.R )
    
    def A(self,gamma,xi,A0=1.):
        return A0*np.exp(-4/gamma * np.sin(xi)**2 )

    def u(self,t,alpha,sigma_c):
        return (t - alpha * sigma_c**2) / (np.sqrt(2) * sigma_c)

    def v(self,t,alpha,sigma_c):
        return alpha*(t - alpha/2 * sigma_c**2)

    def alpha(self,beta,delta):
        return delta - beta**2/4

    def delta(self,gamma,xi,h):
        return 4/gamma * self.c/self.H(h) * np.cos(2 * xi)
    
    def gamma(self,theta):
        return 2*np.sin(theta/2)**2/np.log(2)

    def beta(self,gamma,xi,h):
        return 4/gamma * np.sqrt(self.c/self.H(h)) * np.sin(2*xi)


    def sigma_c(self,sigma_s):
        sigma_p = 0.425 * self.T 
        return np.sqrt(sigma_p**2 + (2*sigma_s/self.c)**2 )

    def pulse(self,t, dim = 1):

        self.dim = dim
        gamma = self.Gamma
        delta = self.delta(gamma,self.xi,self.h)
        beta  = self.beta(gamma,self.xi,self.h)

        if dim == 1:
            alpha = self.alpha(beta,delta)
        else:
            alpha = self.alpha(beta/np.sqrt(2),delta)

        sigma_c = self.sigma_c(self.sigma_s)

        u = self.u(t, alpha, sigma_c)
        v = self.v(t, alpha, sigma_c)

        A = self.A(gamma,self.xi)
        pulse = A*np.exp(-v)*( 1 + erf(u) )
        
        if self.dim == 2:
            alpha = gamma
            u = self.u(t, alpha, sigma_c)
            v = self.v(t, alpha, sigma_c)
            pulse -= A/2*np.exp(-v)*( 1 + erf(u) )

        return pulse

    def pulse_v(self, v, dim = 1):

        self.dim = dim
        gamma = self.gamma(self.theta)

        delta = self.delta(gamma,self.xi,self.h)
        beta  = self.beta(gamma,self.xi,self.h)

        alpha = self.alpha(beta,delta)

        sigma_c = self.sigma_c(self.sigma_s)

        u = np.sqrt(2)*v/(alpha*sigma_c) - alpha*sigma_c/np.sqrt(2)

        A = self.A(gamma,self.xi)
        pulse = A*np.exp(-v)*( 1 + erf(u) )

        return pulse
    
class Pulse:
    def __init__(self, r0 = [0,0,3e6],
                       gane_width=np.deg2rad(1.5),
                       c = 3e8):
        self.c =  c
        self.r0 = r0
        gane_width = np.deg2rad(1.5) # Ширина диаграммы направленности в радианах
        self.gamma = 2*np.sin(gane_width/2)**2/np.log(2)

    def G(self,theta,G0=1):
            # G -- диаграмма направленности
            # theta -- угол падения
            return G0*np.exp(-2/self.gamma * np.sin(theta)**2)
    
    def R(self,x,y,z,x0=0,y0=0,z0=1e6):
        r = np.sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2)
        return r

    def theta_calc(self,x,y,z,x0=0,y0=0,z0=1e6):
        cos_theta = (z0-z)/np.sqrt(
            ( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )
            )
        return np.arccos(cos_theta)

    def local_theta_calc(self,x,y,z,sxx,syy,x0=0,y0=0,z0=1e6):
        nmod = np.sqrt(sxx**2+syy**2+1)
        r    = self.R(x,y,z,x0,y0,z0)
        cos_theta = (sxx*(x-x0) + syy*(y-y0) + (z0 - z))/nmod/r
        return np.arccos(cos_theta)

    def mirror_sort(self,x,y,z,sxx,syy,theta, err = 1):

        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                if  theta[i,j] > np.deg2rad(err):
                    x[i,j]  =None
                    y[i,j]  =None
                    z[i,j]  =None
                    sxx[i,j]=None
                    syy[i,j]=None

                    
        x   = x[~np.isnan(x)]
        y   = y[~np.isnan(y)]
        z   = z[~np.isnan(z)]
        sxx   = sxx[~np.isnan(sxx)]
        syy   = syy[~np.isnan(syy)]
        return x,y,z,sxx,syy

    def E_calc(self,t,omega,timp,R, theta,):

        def A(t,tau,timp):
            if 0 <= t-tau <= timp:
                A = 1
            else:
                A = 0
            return A

        E = np.zeros(R.size)
        c = self.c 
        G = self.G
        tau = R/c
        for i in range(R.size):
            if t - tau[i] < 0:
                E[i] = 0
            else:
                A0 = A(t,tau[i],timp)
                G0 = G(theta[i])
                R0 = R[i]
                E[i] =  A0 * G0 / R0 *\
                    np.exp(
                        1j*omega*(t - tau[i]*np.cos(theta[i]))
                    )
        
        return np.sum(E)**2/2
    




