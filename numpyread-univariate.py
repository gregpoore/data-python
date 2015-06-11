from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from random import randint
import csv
import numpy as np
import matplotlib.pyplot as plt
f=np.genfromtxt(open("CAX_COPD_TRAIN_data.csv","rb"),delimiter=",",skip_header=1) #skiprows=1)
#f=np.loadtxt(open("CAX_COPD_TRAIN_data.csv","rb"),delimiter=",",skiprows=1)
print f.shape
#mat=np.matrix(f)
F1max=0
J,count=0,0
for i in range(1):#range(randint(0,9),100,14):
	train,cv,trainind,cvind=train_test_split(f,range(np.size(f,0)),test_size=0.33,random_state=42 ) #or pass Xand y as parameters to split func
	Xtrain=train[:,2:]
	ytrain=train[:,1]
	E = np.random.uniform(0, 0.1, size=(len(Xtrain), 10))
	X = np.hstack((Xtrain, E))
	y=ytrain
	#plt.figure(1)
	#plt.clf()
	X_indices = np.arange(X.shape[-1])

	selector = SelectPercentile(chi2, percentile=10)
	#selector.fit(X, y)
        
	clf = svm.SVC(kernel='linear')
	#clf.fit(X, y)
	clf_selected = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False)
	#clf_selected.fit(selector.transform(X), y)
        pertile_svm = Pipeline([('percentile', selector), ('svc', clf_selected)])
	pertile_svm.fit(Xtrain,y)
########################################
#This is for understanding how svm predicts good features among all
	#pertile_svm.fit(X,y)
        #print np.shape(X)
	#scores = -np.log10(selector.pvalues_)
	#scores /= scores.max()
	#plt.bar(X_indices - .45, scores, width=.2,label=r'Univariate score ($-Log(p_{value})$)', color='g')
	#svm_weights = (clf.coef_ ** 2).sum(axis=0)
	#svm_weights /= svm_weights.max()

	#plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')
	#svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
	#svm_weights_selected /= svm_weights_selected.max()

	#plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,width=.2, label='SVMweightafterselection', color='b')
	#plt.title("Comparing feature selection")
	#plt.xlabel('Feature number')
	#plt.yticks(())
	#plt.axis('tight')
	#plt.legend(loc='upper right')
	#plt.show()

        Xcv=cv[:,2:]
	ycv=cv[:,1]
	p=int(0)
	n=int(0)
        print np.shape(Xcv)
        #pred=rfe.predict(Xcv)
	pred=pertile_svm.predict(Xcv)
        for pred in pred:
                if pred==1:
			p+=1
		else:
		        n+=1
	print p,n
	p=int(0)
	n=int(0)
        for pred in ycv:
                if pred==1:
			p+=1
		else:
		        n+=1
	print p,n

	#print "pred set before matrix %s" % str(np.shape(pred))
	#print "ycv set before matrix %s" % str(np.shape(ycv))
	#J+=float(f1_score(ycv,pred))
	#count+=1
        #f1score=float(J/count)
#print "No of features %d f1_score %f" % (k,f1score)
        #if f1score>F1max:
	#	F1max=f1score
#		n_features=k
#print F1max
if "":
	f=np.genfromtxt(open("CAX_COPD_TEST_data.csv","rb"),delimiter=",",skiprows=1)
	mat=np.matrix(f)
	Xtest=mat[:,2:]
	p=int(0)
	n=int(0)
if "":
	#in1=np.genfromtxt(open("CAX_COPD_SubmissionFormat.csv","rb"),delimiter=",",skip_header=1)
	#reader=csv.reader(in1)
	out=open("Copy.csv","wb")
	writer=csv.writer(out)
	for row in pred:
	 #   row[1]=row
	    writer.writerow([row])

