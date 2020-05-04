
import numpy as np
from numpy import pi
from scipy import interpolate,integrate
from tqdm import tqdm
# from water.spectrum import Spectrum
from spectrum import Spectrum

class Surface(Spectrum):
    def __init__(self,  N=256, M=100, space='log',
        random_phases = 1, 
        whitening = None, wind = 0,cwm = None, **kwargs):
        print('Начинаю подготовку...')
        Spectrum.__init__(self,**kwargs)
        self.get_spectrum()
        self.N = N
        self.M = M
        KT = self.KT
        self.wind = wind # Направление ветра
        if space=='log':
            self.k = np.logspace(np.log10(KT[0]), np.log10(KT[-1]),self.N + 1)
        else:
            self.k = np.linspace(KT[0], KT[-1],self.N + 1)

        print(\
            "Параметры модели:\n\
               N={},\n\
               M={},\n\
               U={}м/с\n".format(self.N,self.M,self.U10)
            )


        if whitening != None:
            if 'h' in whitening:
                interspace = self.interspace(self.k, N, power=0)
                self.k_heights = self.nodes(interspace,power=0)
                self.k = self.k_heights

            if 's' in whitening:
                interspace = self.interspace(self.k, N, power=2)
                self.k_slopes = self.nodes(interspace,power=2)
                self.k = self.k_slopes

            if 'h' in whitening and 's' in whitening:
                self.k = np.hstack((self.k_heights,self.k_slopes))
                self.k = np.sort(self.k)

        self.phi = np.linspace(0,2*np.pi,self.M + 1)
        self.phi_c = self.phi + self.wind

        # случайные фазы
        if random_phases == 0:
            self.psi = np.array([
                    [0 for m in range(self.M) ] for n in range(self.N) ])
        elif random_phases == 1:
            self.psi = np.array([
                [ np.random.uniform(0,2*pi) for m in range(self.M)]
                            for n in range(self.N) ])


        print('Вычисление высот...')
        self.A = self.amplitude(self.k)
        self.F = self.angle(self.k,self.phi)
        
        print('Вычисление наклонов x...')
        self.A_slopesxx = self.amplitude(self.k,method='xx')
        self.F_slopesxx = self.angle(self.k,self.phi,method='xx')

        print('Вычисление наклонов y...')
        self.A_slopesyy = self.amplitude(self.k,method='yy')
        self.F_slopesyy = self.angle(self.k,self.phi,method='yy')
        print('Подготовка завершена.')

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
        # N = self.N
        N = self.N
        # print(k.size)
        if method =='h':
            Phi = lambda phi,k: self.Phi(k,phi)
        elif method == 'xx':
            Phi = lambda phi,k: self.Phi(k,phi)*np.cos(phi)**2
        elif method == 'yy':
            Phi = lambda phi,k: self.Phi(k,phi)*np.sin(phi)**2
        else:
            Phi = lambda phi,k: self.Phi(k,phi)

        integral = np.zeros((N,M))
        # print(integral.shape)
        for i in range(N):
            for j in range(M):
                # integral[i][j] = integrate.quad( Phi, phi[j], phi[j+1], args=(k[i],) )[0]
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
        # progress_bar = tqdm( total = N-1)
        for i in range(1,N):
            integral[i-1] = integrate.quad(S,k[i-1],k[i])[0] 
        #     progress_bar.update(1)
        # progress_bar.clear()
        # progress_bar.close()
        # integral = np.array([ integrate.quad(S,k[i-1],k[i])[0] for i in range(1,N) ])
        amplitude = np.sqrt(2 *integral )
        return np.array(amplitude)


    
    def calc_amplitude(self,k,phi):
        N = k.size
        M = phi.size 
        # print(N,M)
        integral = np.zeros((N-1,M-1))
        progress_bar = tqdm(total=(N-1)*M)
        spectrum_k = lambda k: self.spectrum(k)
        spectrum_phi = lambda k,phi: self.Phi(k,phi)
        func = lambda k,phi: spectrum_k(k)*spectrum_phi(k,phi)
        for i in range(1,N):
            for j in range(1,M):
                integral[i-1][j-1] = integrate.dblquad(func,phi[j-1],phi[j],lambda x: k[i-1],lambda x: k[i] )[0]
            progress_bar.update(1)
        progress_bar.clear()
        progress_bar.close()
        return np.sqrt(2*integral)


    def heights(self,r,t,method='h'):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi
        try:
            A = self.A
            F = self.F
        except:
            A = self.amplitude(k,method=method)
            F = self.angle(k,phi,method=method)
        psi = self.psi
        self.surface = 0
        # self.amplitudes = np.array([ A[i]*sum(F[i])  for i in range(N)])
        progress_bar = tqdm( total = N*M,  leave = False )
        for n in range(N):
            for m in range(M):
                self.surface += A[n] * \
                np.cos(
                    +k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                    +psi[n][m]
                    +self.omega_k(k[n])*t) \
                    * F[n][m]
                progress_bar.update(1)
        progress_bar.close()
        progress_bar.clear()
        print()
        return self.surface

    def slopesxx(self,r,t):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi

        try:
            A = self.A_slopesxx
            F = self.F_slopesxx
        except:

            self.A_slopesxx = self.amplitude(k,method='xx')
            self.F_slopesxx = self.angle(k,phi,method='xx')

            A = self.A_slopesxx
            F = self.F_slopesxx

        psi = self.psi
        self.surface = 0
        progress_bar = tqdm( total = N*M,  leave = False )
        for n in range(N):
            for m in range(M):
                self.surface += A[n] * \
                np.cos(
                    +k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                    +psi[n][m]
                    +self.omega_k(k[n])*t) \
                    * F[n][m]
                progress_bar.update(1)
        progress_bar.close()
        progress_bar.clear()
        print()
        return self.surface

    def slopesyy(self,r,t):
        N = self.N
        M= self.M
        k = self.k
        phi = self.phi

        try:
            A = self.A_slopesyy
            F = self.F_slopesyy
        except:

            self.A_slopesyy = self.amplitude(k,method='yy')
            self.F_slopesyy = self.angle(k,phi,method='yy')

            A = self.A_slopesyy
            F = self.F_slopesyy

        psi = self.psi
        self.surface = 0
        progress_bar = tqdm( total = N*M,  leave = False )
        for n in range(N):
            for m in range(M):
                self.surface += A[n] * \
                np.cos(
                    +k[n]*(r[0]*np.cos(phi[m])+r[1]*np.sin(phi[m]))
                    +psi[n][m]
                    +self.omega_k(k[n])*t) \
                    * F[n][m]
                progress_bar.update(1)
        progress_bar.close()
        progress_bar.clear()
        print()
        return self.surface
