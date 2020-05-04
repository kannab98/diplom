
import matplotlib.pyplot as plt
from numpy.random import uniform
x = uniform(-5000, 5000, size=(10,10))
y = uniform(-5000, 5000, size=(10,10))
plt.scatter(x,y)
plt.show()