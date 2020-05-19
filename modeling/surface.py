
import numpy as np
import configparser
from numpy import pi
from scipy import interpolate,integrate
from tqdm import tqdm
from numba import jit,njit, prange
# from water.spectrum import Spectrum
from spectrum import Spectrum


class Surface(Spectrum):
    def __init__(self, 
            conf_file = None,  N = None, M = None, 
            random_phases = None, kfrag = None, 
            whitening = None, wind = None, 
            custom_spectrum = None,
            **kwargs):



        if custom_spectrum == None:
            Spectrum.__init__(self,conf_file, **kwargs)
            self.spectrum = self.get_spectrum()
            KT = self.KT
        else:
            self.spectrum = custom_spectrum.get_spectrum()
            KT = custom_spectrum.KT
            self.KT = KT
            self.U10 = custom_spectrum.U10

        if conf_file != None:
            config = configparser.ConfigParser()
            config.read(conf_file)
            config = config['Surface']




        if N == None:
            try:
                self.N = int(config['N'])
            except:
                self.N = 256
        else:
            self.N = N

        if M == None:
            try:
                self.M = int(config['M'])
            except: 
                self.M = 128
        else:
            self.M = M



        if wind == None:
            try:
                self.wind = float(config['WindDirection'])
                self.wind = np.deg2rad(self.wind)
            except:
                self.wind = 0
        else:
            self.wind = np.deg2rad(wind)

        if kfrag == None:
            try:
                kfrag  = config['WaveNumberFragmentation']
            except:
                kfrag = 'log'

        if kfrag == 'log':
            self.k = np.logspace(np.log10(self.k_m/4), np.log10(self.k_edge['Ku']), self.N + 1)
            self.k = self.k[ np.where(self.k <= self.k_edge[self.band]) ]
            self.N = self.k.size - 1
        else:
            self.k = np.linspace(KT[0], KT[-1],self.N + 1)

        print(\
            "Параметры модели:\n\
                N={},\n\
                M={},\n\
                U={} м/с,\n\
                Band={}".format(self.N,self.M,self.U10,self.band)
            )

        if whitening == None:
            try:
                whitening = int(config['SpectrumWhitening'])
            except:
                whitening = 0

        if whitening != 0:
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


        if random_phases == None:
            try:
                random_phases = int(config['RandomPhases'])
            except:
                random_phases = 1

        if random_phases == 0:
            self.psi = np.array([
                    [0 for m in range(self.M) ] for n in range(self.N) ])
        elif random_phases == 1:
            self.psi = np.array([
                [ np.random.uniform(0,2*pi) for m in range(self.M)]
                            for n in range(self.N) ])

        print(\
            "Методы:\n\
                Случайные фазы     {}\n\
                Отбеливание        {}\n\
                Заостренная волна  {}\n\
            ".format(bool(random_phases), bool(whitening), False)
            )
                            


        # if heights == 1:
        #     print('Вычисление высот...')
        #     self.A = self.amplitude(self.k)
        #     self.F = self.angle(self.k,self.phi)
        
        # if slopes == 1:
        #     print('Вычисление полных наклонов...')
        #     self.A_slopes = self.amplitude(self.k,method='s')
        #     self.F_slopes = self.angle(self.k,self.phi,method='s')

        # if slopesxx == 1:
        #     print('Вычисление наклонов x...')
        #     self.A_slopesxx = self.amplitude(self.k,method='xx')
        #     self.F_slopesxx = self.angle(self.k,self.phi,method='xx')

        # if slopesyy == 1:
        #     print('Вычисление наклонов y...')
        #     self.A_slopesyy = self.amplitude(self.k,method='yy')
        #     self.F_slopesyy = self.angle(self.k,self.phi,method='yy')


        # print('Подготовка завершена.')

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


    def angle(self, k, phi, calc='heights'):
        M = self.M
        N = self.N
        if calc=='heights':
            Phi = lambda phi,k: self.Phi(k,phi)
        elif calc== 'slopesxx':
            Phi = lambda phi,k: self.Phi(k,phi)*np.cos(phi)**2
        elif calc== 'slopesyy':
            Phi = lambda phi,k: self.Phi(k,phi)*np.sin(phi)**2
        else:
            Phi = lambda phi,k: self.Phi(k,phi)

        integral = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                integral[i][j] = np.trapz( Phi( phi[j:j+2],k[i] ), phi[j:j+2])
        amplitude = np.sqrt(2 *integral )
        return amplitude

    def amplitude(self, k, calc = 'heights'):
        N = len(k)
        if calc == 'heights':
            S = self.spectrum
        else:
            S = lambda k: self.spectrum(k) * k**2

        integral = np.zeros(k.size-1)
        for i in range(1,N):
            integral[i-1] = integrate.quad(S,k[i-1],k[i])[0] 
        amplitude = np.sqrt(2 *integral )
        return np.array(amplitude)


    def data(self):
        k = self.k
        phi = self.phi
        psi = self.psi
        psi = self.psi

        A_h = self.amplitude(k, calc = 'heights')
        # A_s = self.amplitude(k, calc = 'slopes')
        A_s = k[0:-1]**2*self.amplitude(k, calc = 'heights')

        F_h = self.angle(k, phi, calc = 'heights')
        # F_sxx = self.angle(k, phi, calc = 'slopesxx') 
        F_sxx = np.cos(phi[0:-1])**2 * F_h
        F_syy = np.sin(phi[0:-1])**2 * F_h
        # F_syy = self.angle(k, phi, calc = 'slopesxx' )
        print('Подготовка завершена.')

        return  k, phi, psi, A_h, A_s, F_h, F_sxx, F_syy


# surface = Surface(1,0,0,0, N=2048, M=1024, wind=0,random_phases=0)

# k = surface.k
# phi = surface.phi
# A = surface.A
# F = surface.F
# psi = surface.psi
# data  = pd.DataFrame({'k':k[0:-1],  'A':A})
# data.to_csv(r'kA.tsv', sep='\t',index=False)
# data  = pd.DataFrame({'phi':phi})
# data.to_csv(r'phi.tsv', sep='\t',index=False)
# data  = pd.DataFrame({'F':F.flatten(),'psi':psi.flatten()})
# data.to_csv(r'Fpsi.tsv', sep='\t',index=False)