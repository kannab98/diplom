import sys
import os    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.insert(0, '../scripts')
from radiolocator import Radiolocator


file_name =  os.path.basename(sys.argv[0])
(file, ext) = os.path.splitext(file_name)


radiolocator = Radiolocator(xi = 0, sigma = 2)
t = np.linspace(-1.5e-7, 6e-7, 256)
pulse = radiolocator.pulse(t)     
data = pd.DataFrame({"col1":t,"col2":pulse})
data.to_csv(file+r'.tsv',index=False,header=None,sep='\t')
plt.plot(t,pulse)
plt.show()