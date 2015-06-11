import numpy as np
from numpy import *
import math
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    sigma=np.matrix(np.diag(sigma))
    print size,len(mu),sigma
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I        
        result = math.pow(math.e, -0.5 * (x_mu.T * inv * x_mu))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
#    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
#    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
#    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
#    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
#    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).dot(np.linalg.inv(cov))).dot((x-mu).T)
    return float(part1 * np.exp(part2))

def test_gauss_pdf():
    x = np.array([[0],[0]])
    mu  = np.array([[0],[0]])
    cov = np.eye(2) 

    print(pdf_multivariate_gauss(x, mu, cov))

    # prints 0.15915494309189535

def estimategauss(X):
        sigma2=[0.0]*np.size(X,1)
        mu=[0.0]*np.size(X,1)
        sigma=np.matrix([[0]*np.size(X,1) for i in range(np.size(X,0))])
#sigma=[0]*np.size(X,1)
        for i in range(np.size(X,1)):
                mu[i]=float(np.sum(X[:,i]))/float(np.size(X,0))
        #mu=np.matrix(mu)
        #for i in range(np.size(X,0)):
        #       sigma[i,:]=np.power((X[i,:]-mu),2)
        sigma=np.power(X-mu,2)
        for i in range(np.size(X,1)):
                sigma2[i]=float(np.sum(sigma[:,i])/np.size(X,0))
        return {'mu':mu,'sigma2':sigma2}


def multivariategauss(X,mu,sigma2):
        k=len(mu)
        sigma2=np.matrix(np.diag(sigma2))
        #if (np.size(sigma2,0)==1):
        #       sigma2=np.diag(sigma2)
        x_mu=X-mu
        p=np.power(2*np.pi,-(k/2))*np.power(np.linalg.det(sigma2),-0.5)*\
         np.exp(-0.5*(x_mu).dot(np.linalg.inv(sigma2)).dot(x_mu.T))
        ret=np.diagonal(p)
        return ret 

if __name__ == '__main__':
#    test_gauss_pdf()                           
	x=np.matrix([[2, 2, 2, 2],
		[3, 3, 3, 3]])
#,[2, 2, 2],
#		[3, 3, 3]])
	a=estimategauss(x)
        p=multivariategauss(x,a['mu'],a['sigma2'])
	#p=pdf_multivariate_gauss(x,np.matrix(a['mu']).T,np.diag(a['sigma2']))
        #p= norm_pdf_multivariate(np.transpose(x), np.reshape(a['mu'],(len(a['mu']),1)), np.array(a['sigma2'])) 
        print p
