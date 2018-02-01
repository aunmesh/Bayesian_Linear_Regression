from numpy import genfromtxt
import numpy as np
from BLR import BLR
import matplotlib.pyplot as plt
import sys

np.random.seed(60)

fname = 'blrdata.txt'
my_data = genfromtxt( fname , delimiter=',')

d_size = int(sys.argv[2])

X = my_data[:d_size,:2]
Y = my_data[:d_size,2]

Lambda = 2
Beta = 25

Model = BLR(X,Y ,np.ones(2) , Lambda, Beta)
Model.InferPosterior()

N = 10000
Vals = Model.Posterior.Generate(N)


# Fixing random state for reproducibility

colors = np.random.rand(N)
#area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
area = float(sys.argv[1])
#plt.scatter( Vals[:,0], Vals[:,1], s=area, c=colors, alpha=0.5)
plt.scatter( Vals[:,0], Vals[:,1], s=area, alpha=int(sys.argv[3]))

plt.show()
