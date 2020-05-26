from numpy import *
from matplotlib.pyplot import *
xmin = 0.1
xmax = 100
N = 1000
a = logspace(log10(xmin), log10(xmax),N)
b = zeros(a.size)
b[0] = a[0]

for i in range(1, b.size):
    b[i] = b[i-1]*xmax/(N-1)

loglog(a,a)
loglog(b,b)
show()