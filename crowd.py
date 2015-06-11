from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from random import randint
import csv
import numpy as np
import matplotlib.pyplot as plt
f=np.genfromtxt(open("CAX_COPD_TRAIN_data.csv","rb"),delimiter=",",skip_header=1) #skiprows=1)
F1max=0
J,count=0,0
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

def selectThreshold(pval,yval):
	bestf1=0
	f1=0
	epsilon=0
	step=(float(max(pval))-float(min(pval)))/1000.0
	for i in np.arange(min(pval),max(pval),step):
		pred=(pval<i)
		f1=float(f1_score(yval,pred))
		if f1>bestf1:
		    bestf1=f1
		    epsilon=i
        return epsilon


if __name__ == '__main__':
	#Xtrain=f[:,2:]
        #gaussian=[3,4,5,7,21,23,24,25,26,28,28,30,32,36,37,60]
#for i in range(randint(0,9),100,14):
	train,cv,trainind,cvind=train_test_split(f,range(np.size(f,0)),test_size=0.33,random_state=42 ) #or pass Xand y as parameters to split func
        gaussian=[3,4,5,21,23,25,28,32]
        Xtrain=train[:,gaussian]
	ytrain=train[:,1]
	a=estimategauss(Xtrain)
	Xcv=cv[:,gaussian]
	ycv=cv[:,1]
	pval=multivariategauss(Xtrain,a['mu'],a['sigma2'])
        print len(pval),min(pval),max(pval)
	epsilon=selectThreshold(pval,ytrain)
        print epsilon
	p=multivariategauss(Xcv,a['mu'],a['sigma2'])
	pred=(p<epsilon)
	p=int(0)
	n=int(0)
	for x in pred:
	    if x==1:
		p+=1
	    else:
		n+=1
	print "Predicted pos %d neg %d" %(p,n)
	p=int(0)
	n=int(0)
	for x in ycv:
	    if x==1:
		p+=1
	    else:
		n+=1
	print "Actual pos %d neg %d" %(p,n)
#	J+=float(f1_score(ycv,pred))
#	count+=1
#if f1score>F1max:
#	F1max=f1score
#	n_features=k
if "":
	f=np.genfromtxt(open("CAX_COPD_TEST_data.csv","rb"),delimiter=",",skip_header=1)
	mat=np.matrix(f)
	Xtest=mat[:,gaussian]
	p=multivariategauss(Xtest,a['mu'],a['sigma2'])
	pred=(p<epsilon)
	p=int(0)
	n=int(0)
	for x in pred:
	    print x
	    if x==1:
		p+=1
	    else:
		n+=1
	print p,n
if "":	
	for x in pred:
	    if x==1:
		p+=1
	    else:
		n+=1
	print "Predicted pos %d neg %d" %(p,n)
	p=int(0)
	n=int(0)
	for x in ycv:
	    if x==1:
		p+=1
	    else:
		n+=1
	print "Actual pos %d neg %d" %(p,n)
#if "":
if "":
	#in1=np.genfromtxt(open("CAX_COPD_SubmissionFormat.csv","rb"),delimiter=",",skip_header=1)
	#reader=csv.reader(in1)
	out=open("Copy.csv","wb")
	writer=csv.writer(out)
	for row in pred:
	 #   row[1]=row
	    writer.writerow([row])

