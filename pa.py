from numpy import genfromtxt
import numpy as np
from BLR import BLR
import matplotlib.pyplot as plt
import sys

np.random.seed(60)

fname = 'blrdata.txt'
my_data = genfromtxt( fname , delimiter=',')

sizes = [20,10,5,2,1]

#d_size = int(sys.argv[2])

#List for storing 10 
def LinearEq( X, W ):
    Vals = []

    for x in X:
        Vals.append( np.dot( W, [1, x]) )

    return Vals    


temp = 0

for d_size in sizes:
    temp+=1

    X = my_data[:d_size,:2]
    Y = my_data[:d_size,2]
    
    Lambda = 2
    Beta = 25
    
    Model = BLR(X,Y ,np.ones(2) , Lambda, Beta)
    Model.InferPosterior()
    
    N = 10000
    Vals = Model.Posterior.Generate(N)  

    # Fixing random state for reproducibility

    plt.figure(temp + 10)
    colors = np.random.rand(N)
    #area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
    area = float(sys.argv[1])
    #plt.scatter( Vals[:,0], Vals[:,1], s=area, c=colors, alpha=0.5)
    plt.scatter( Vals[:,0], Vals[:,1], s=area, alpha=int(sys.argv[2]))
    plt.savefig('ScatterPlot_'+ str(d_size)+ '.png', bbox_inches='tight')

    W_s = Vals[0:10]
    X = [-1, -0.5, 0, 0.5, 1]

    for W in W_s:
        Y = LinearEq(X,W)
        plt.figure(temp)
        plt.plot(X, Y)
           
        
    #plt.plot([1,2,3,4], [1,4,9,16], 'ro')
    #plt.axis([0, 6, 0, 20])
    plt.savefig('PredictivePlot_'+str(d_size)+'.png', bbox_inches='tight')

    del(Model)

#plt.show()
