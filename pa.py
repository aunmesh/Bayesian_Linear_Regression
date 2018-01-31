from numpy import genfromtxt
import numpy as np
import BLR

fname = 'blrdata.txt'
my_data = genfromtxt( fname , delimiter=',')

X = my_data[:,:2]
Y = my_data[:,2]

Lambda = 2
Beta = 25

Model = BLR.BLR(X,Y ,np.ones(2) , Lambda, Beta)
