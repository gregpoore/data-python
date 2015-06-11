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
#f=np.loadtxt(open("CAX_COPD_TRAIN_data.csv","rb"),delimiter=",",skiprows=1)
print f.shape
#mat=np.matrix(f)
F1max=0
for k in range(50):
	J,count=0,0
	for i in range(randint(0,9),100,14):
		train,cv,trainind,cvind=train_test_split(f,range(np.size(f,0)),test_size=0.33,random_state=i ) #or pass Xand y as parameters to split func
		Xtrain=train[:,2:]
		ytrain=train[:,1]
		#print trainind,cvind
		#itrain=mat[:1100,0]
		#for x in X:
		 #  print x
		model = LogisticRegression()
		#from sklearn.svm import SVC
		#model=SVC(kernel='linear')
		#rfe = RFE(model, k)
		#rfe = rfe.fit(Xtrain, ytrain)
                clf=SelectKBest(chi2, k=k).fit_transform(Xtrain, ytrain)
		#print(rfe.support_)
		#print(rfe.ranking_) This is one of the results expected
		Xcv=cv[:,2:]
		ycv=cv[:,1]
		p=int(0)
		n=int(0)
		#pred=rfe.predict(Xcv)
		pred=clf.predict(Xcv)
		#print "pred set before matrix %s" % str(np.shape(pred))
		#print "ycv set before matrix %s" % str(np.shape(ycv))
		J+=float(f1_score(ycv,pred))
		count+=1
        f1score=float(J/count)
	print "No of features %d f1_score %f" % (k,f1score)
        if f1score>F1max:
		F1max=f1score
		n_features=k
print n_features,F1max
Xtrain=f[:,2:]
ytrain=f[:,1]
model = LogisticRegression()
rfe = RFE(model, n_features)
rfe = rfe.fit(Xtrain, ytrain)
if "":
        predm,ycvm=np.transpose(np.matrix(pred)),np.transpose(np.matrix(ycv))
        tp=np.sum((predm==1)&(ycvm==1));
        fp=np.sum((predm==1)&(ycvm==0));
        fn=np.sum((predm==0)&(ycvm==1));
        if(((tp+fp)==0)|((tp+fn)==0)):
		prec=-1;
		recall=-1;
	else:
		prec=tp/(tp+fp);
		recall=tp/(tp+fn);
	F1=(2*prec*recall)/(prec+recall);
        print "F1 score is %f" % float(F1)
if "":
	print "pred set %s" % str(np.shape(predm))
	print "ycv set %s" % str(np.shape(ycvm))
        print np.shape(ycvm),np.size(ycvm,0)
        if np.size(ycvm,0)>0:
		J=(1/np.size(ycvm,0))*((-np.transpose(ycvm)*np.log(predm))-(np.transpose(1-ycvm)*np.log(1-predm)))
		print float(J) #Cost function useful if we write our own hypothesis 
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
f=np.genfromtxt(open("CAX_COPD_TEST_data.csv","rb"),delimiter=",",skiprows=1)
mat=np.matrix(f)
Xtest=mat[:,2:]
p=int(0)
n=int(0)
pred=rfe.predict(Xtest)
for x in pred:
    print x
    if x==1:
	p+=1
    else:
	n+=1
print p,n
if "":
	with open('CAX_COPD_SubmissionFormat.csv','r') as csvinput:
	    with open('Coutput.csv', 'w') as csvoutput:
		writer = csv.writer(csvoutput, lineterminator='\n')
		reader = csv.reader(csvinput)

		all = []
		row = next(reader)
		#row.append('Berry')
		all.append(row)

		for row in reader :
		    row.append(pred)
		    all.append(row)

		writer.writerows(all)
if "":
	#in1=np.genfromtxt(open("CAX_COPD_SubmissionFormat.csv","rb"),delimiter=",",skip_header=1)
	#reader=csv.reader(in1)
	out=open("Copy.csv","wb")
	writer=csv.writer(out)
	for row in pred:
	 #   row[1]=row
	    writer.writerow([row])

