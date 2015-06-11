import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from random import randint
import csv
import numpy as np
import matplotlib.pyplot as plt
f=np.genfromtxt(open("CAX_COPD_TRAIN_data.csv","rb"),delimiter=",",skip_header=1) #skiprows=1)
print f.shape
def categorize(train_data,test_data,train_class,n_features):
    #cf= ExtraTreesClassifier()
    #cf.fit(train_data,train_class)
    #print (cf.feature_importances_)
    
    #lsvmcf = sklearn.svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=100.0)  
    model = LogisticRegression()
    lgr = LogisticRegression(C=100.0,penalty='l1')    
    #knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=10, p=2, metric='minkowski', metric_params=None)
    svmlcf = sklearn.svm.SVC(C=1000.0, kernel='linear', degree=1, gamma=0.01,  probability=True)#2
    svmcf = sklearn.svm.SVC(C=1000.0, kernel='rbf', degree=1, gamma=0.01,  probability=True)#2
    cf = DecisionTreeClassifier() 
    dct = DecisionTreeClassifier(criterion='gini', splitter='best',  min_samples_split=7, min_samples_leaf=4)
    rf = RandomForestClassifier(n_estimators=10, criterion='gini',  min_samples_split=7, min_samples_leaf=4, max_features='auto')
    gnb = GaussianNB()  #1
    adbst = sklearn.ensemble.AdaBoostClassifier(base_estimator=rf, n_estimators=5, learning_rate=1.0, algorithm='SAMME.R', random_state=True)

    #ch2 = SelectKBest(chi2, k=n_features)
    #train_data = ch2.fit_transform(train_data, train_class)
    #test_data = ch2.transform(test_data)

    #rfe = RFE(svmlcf,n_features)
    #rfe = rfe.fit(train_data, train_class)
    gnb.fit(train_data,train_class)
    return gnb.predict(test_data)

F1max=0
binary=[2,6,8,10,12,14,15,17,38,41,47,49,51,52]
gaussian=[3,4,5,7,21,23,24,25,26,28,30,32,34,35,36,37,60]
#gaussian=[3,4,5,21,23,25,28,32]
for j in range(50,1,-1):
	J,count=0,0
	for i in range(randint(0,9),100,14):
		train,cv,trainind,cvind=train_test_split(f,range(np.size(f,0)),test_size=0.33,random_state=i ) #or pass Xand y as parameters to split func
#Try separating only gaussian features
                Xtrain=train[:,gaussian]
		#Xtrain=train[:,2:]
		ytrain=train[:,1]
		#Xcv=cv[:,2:]
		Xcv=cv[:,gaussian]
		ycv=cv[:,1]
                Xb=train[:,binary]
                mod=LogisticRegression()
                Xb=mod.fit_transform(Xb,ytrain)
                Xtrain=np.c_[Xtrain,Xb]
                Xcvb=cv[:,binary]
                Xcvb=mod.transform(Xcvb)
                Xcv=np.c_[Xcv,Xcvb]
		p=int(0)
		n=int(0)
		pred=categorize(Xtrain,Xcv,ytrain,j)
		J+=float(f1_score(ycv,pred))
		count+=1
	f1score=float(J/count)
	print "No of feature %d f1_score %f" % (j,f1score)
	if f1score>F1max:
		F1max=f1score
                n_features=j
print F1max,n_features
#Xtrain=f[:,gaussian]
#Xtrain=f[:,2:]
ytrain=f[:,1]

#if "":
f=np.genfromtxt(open("CAX_COPD_TEST_data.csv","rb"),delimiter=",",skip_header=1)
mat=np.matrix(f)
Xtest=mat[:,gaussian]
Xb=mat[:,binary]
#mod=LogisticRegression()
Xb=mod.transform(Xb)#,ytrain)
Xtest=np.c_[Xtest,Xb]
#Xtest=mat[:,2:]
p=int(0)
n=int(0)
#n_features=10
pred=categorize(Xtrain,Xtest,ytrain,n_features)
print n_features
for x in pred:
    #print x
    if x==1:
	p+=1
    else:
	n+=1
print p,n
#if "":
out=open("Copy.csv","wb")
writer=csv.writer(out)
for row in pred:
    writer.writerow([row])

