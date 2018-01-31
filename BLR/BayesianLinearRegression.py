import numpy as np
import sys
from numpy.linalg import inv,det

#sys.path.append("../")

from Distributions.Gaussian import MultGaussian as MG

class BLR():


    '''
    Constructor for Bayesian Linear Regression

    Args:
        X - Data Matrix of size N*D. N is the column size(No. of data points)
        Y - Label Vector of size N.
        Lambda - Lambda is the precision of the covariance Matrix of Prior
        Beta - Precision of the Likelihood Model

    '''
    def __init__(self, X, Y, W_initial, Lambda, Beta):


        self.X = X * 1.0
        self.Y = Y
        self.W_initial = W_initial * 1.0
        self.Lambda = Lambda * 1.0
        self.Beta = Beta * 1.0

        self.W_mean = np.zeros(len(W_initial))
        self.W_covar = (1.0 / self.Lambda) * np.eye(len(W_initial))

        self.Prior = MG([self.W_mean,self.W_covar])
        self.Posterior = None
        
    '''
    Function for Inferring the posterior of the Model from the 
    Prior and the Likelihood.

    Args:
        X - Data Matrix size (N * D, N is the no. of data points)
        Y - Label Vector of Size N
        Lambda - Precision of the Prior
        Beta - Precision of the Likelihood

    '''    
    #def InferPosterior(self, X,Y,Lambda,Beta):
    def InferPosterior(self):

        DcrossD = np.matmul( np.transpose(self.X) , self.X )
        ID = np.eye(self.X.shape[1])
        
        Post_Covar = inv( self.Beta * DcrossD  + self.Lambda * ID )

        Temp = inv( DcrossD + (self.Lambda/self.Beta) * ID )
        Temp2 = np.matmul( Temp , np.transpose(self.X) )

        Post_Mean = np.matmul(Temp2, self.Y)

        self.Posterior = MG( [Post_Mean, Post_Covar]  )
        

                        
    

        

                    
