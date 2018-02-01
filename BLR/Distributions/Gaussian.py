import numpy as np
from Distributions import Distribution
from numpy.linalg import inv,det

class MultGaussian(Distribution):

    #Params = [Mean,Var]
    def __init__(self, Params):

        assert(type(Params) == list),'Pass the parameters in a list [mean,CoVarMatrix]'
        assert(type(Params[0]) == np.ndarray and type(Params[0]) == np.ndarray ),'Mean vector and Covariance Matrix must be a Numpy array'

        assert(len(Params) == 2),'Give 2 parameters Mean and Variance.'
        assert( len(Params[0]) > 1 ),'Univariate Gaussians are to be instantiated using different class'

        #Implement check for square matrix
        print("WARNING: Positive semi definite check for Covariance Matrix has not been implemented.")

        super(MultGaussian,self).__init__(Params)
        self.dims = len(Params[0])

        #Inverse of Covariance Matrix
        self.cov_inv = inv(1.0 * self.Params[1])
        self.det_cov = det( 2 * np.pi * self.Params[1])
                

    def Probab(self,input):
        assert(len(input) == len(self.Params[0])),'Input Dimensionality should be of size %s' %(str(self.dims))

        diff = input*1.0 - self.Params[0]*1.0

        temp = np.matmul(diff,self.cov_inv)
        mult = -0.5 *  np.matmul(temp,np.transpose(diff))
         
        return  ( 1.0 / self.det_cov ) * np.exp(mult)

    #Returns N samples generated from this Distribution
    def Generate(self,N):
	return np.random.multivariate_normal(self.Params[0], self.Params[1], 1000)

