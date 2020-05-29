
import numpy as np
from numpy import pi
from scipy import interpolate,integrate
from spectrum import Spectrum


class Surface(Spectrum):
    def __init__(self,**kwargs):


        Spectrum.__init__(self,conf_file, **kwargs)
        self.spectrum = self.get_spectrum()
        self.k = np.logspace(np.log10(self.k_m/4), np.log10(self.k_edge['Ku']), self.N + 1)
        self.phi = np.linspace(0,2*np.pi,self.M + 1)

        self.psi = np.array([[ np.random.uniform(0,2*pi) for m in range(self.M)]
                            for n in range(self.N) ])


        print('Вычисление высот...')
        self.A = self.amplitude(self.k)
        self.F = self.angle(self.k,self.phi)
        
        print('Вычисление полных наклонов...')
        self.A_slopes = self.amplitude(self.k,method='s')
        self.F_slopes = self.angle(self.k,self.phi,method='s')

        print('Вычисление наклонов x...')
        self.A_slopesxx = self.amplitude(self.k,method='xx')
        self.F_slopesxx = self.angle(self.k,self.phi,method='xx')

        print('Вычисление наклонов y...')
        self.A_slopesyy = self.amplitude(self.k,method='yy')
        self.F_slopesyy = self.angle(self.k,self.phi,method='yy')


    def B(self,k):
          def b(k):
              b=(
                  -0.28+0.65*np.exp(-0.75*np.log(k/self.k_m))
                  +0.01*np.exp(-0.2+0.7*np.log10(k/self.k_m))
                )
              return b
          B=10**b(k)
          return B

    def Phi(self,k,phi):
        # Функция углового распределения
        phi = phi -self.wind
        normalization = lambda B: B/np.arctan(np.sinh(2* (pi)*B))
        B0 = self.B(k)
        A0 = normalization(B0)
        Phi = A0/np.cosh(2*B0*(phi) )
        return Phi


    def angle(self,k,phi,method='h'):
        M = self.M
        N = self.N
        if method =='h':
            Phi = lambda phi,k: self.Phi(k,phi)
        elif method == 'xx':
            Phi = lambda phi,k: self.Phi(k,phi)*np.cos(phi)**2
        elif method == 'yy':
            Phi = lambda phi,k: self.Phi(k,phi)*np.sin(phi)**2
        else:
            Phi = lambda phi,k: self.Phi(k,phi)

        integral = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                integral[i][j] = np.trapz( Phi( phi[j:j+2],k[i] ), phi[j:j+2])
        amplitude = np.sqrt(2 *integral )
        return amplitude

    def amplitude(self, k,method='h'):
        N = len(k)
        if method == 'h':
            S = self.spectrum
        else:
            S = lambda k: self.spectrum(k) * k**2
        integral = np.zeros(k.size-1)
        for i in range(1,N):
            integral[i-1] = integrate.quad(S,k[i-1],k[i])[0] 
        amplitude = np.sqrt(2 *integral )
        return np.array(amplitude)


