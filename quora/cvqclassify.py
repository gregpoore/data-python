import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
def normalise(traindata,testdata):
    nrml=sklearn.preprocessing.MinMaxScaler()
    nrml.fit(traindata)
    Xtrain=nrml.transform(traindata)
    Xtest=nrml.transform(testdata)
    return Xtrain,Xtest
    
fp=open("trainingdata1.txt","rU")
m,n=map(int,fp.readline().split(' '))
print m
#Revert it
#m=166
print n
cl=[]
iden=[]
inp=[]
x=[]
fvec=[]
feature={}
entry=[]
for i in range(m):
    x=fp.readline().strip().split(' ')
    iden.append(x[0])
    cl.append(x[1])
    inp.append(x[2:])
    entry=x[2:]
    #print a 
#for entry in inp:
    for j in range(n):
        key,val=entry[j].split(':')
        feature[key]=val
    feature={int(key):float(val) for key,val in feature.items()}
    fvec.append(feature.copy())
#for i in range(m):
    row=[]
    for key in sorted(fvec[i]):
	row.append(fvec[i][key])
    row=np.matrix(row)
    if i==0:
        train=row
    else:
        train=np.r_[train,row]

orig=['-1','+1']
orig2label={orig[i]:i for i in range(2)}
ytrain=[orig2label[i] for i in cl]

Xtrain,Xcv,trainlabel,cvlabel=train_test_split(train,ytrain,test_size=0.33,random_state=14 ) #or pass Xand y as parameters to split func
Xtrain,Xcv=normalise(Xtrain,Xcv)
print "shapes of matrices"
print np.shape(Xtrain)
print np.shape(Xcv)
#Actual fitting and prediction
rf = RandomForestClassifier(n_estimators=10, criterion='gini',  min_samples_split=7, min_samples_leaf=4, max_features='auto')
adbst = sklearn.ensemble.AdaBoostClassifier(base_estimator=rf, n_estimators=5, learning_rate=1.0, algorithm='SAMME.R', random_state=True)
cf = adbst
#train_data = train_data[:,(0,1,3,4,10,11,19,20)]
#test_data = test_data[:,(0,1,3,4,10,11,19,20)]
cf.fit(Xtrain,trainlabel)
pred=cf.predict(Xcv)
print cf.score(Xtrain,trainlabel)
print cf.score(Xcv,cvlabel)
print pred
#model=LogisticRegression()
#rfe=RFE(model,21)
#rfe=rfe.fit(Xtrain,trainlabel)
#print(rfe.support_)
#print(rfe.ranking_) #This is one of the results expected
#predcv=rfe.predict(Xcv)
#print rfe.score(Xtrain,trainlabel)
#print rfe.score(Xcv,cvlabel)
#print predcv

if "":
	pca=PCA(n_components=2)
	#kbest=SelectKBest(chi2,k=10)
	print pca
	clf=SVC(kernel="linear")
	kbest_log=Pipeline([("features",pca),("clf",clf)])
	kbest_log.fit(Xtrain,trainlabel)
	predcv=kbest_log.predict(Xcv)
	print predcv
if "":
	#Feature union
	# This dataset is way to high-dimensional. Better do PCA:
	pca = PCA(n_components=2)
	# Maybe some original features where good, too?
	#selection = SelectKBest(k=1)
	# Build estimator from PCA and Univariate selection:
	combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
	# Use combined features to transform dataset:
	X_features = combined_features.fit(Xtrain, trainlabel).transform(Xtrain)
	svm = SVC(kernel="linear")
	# Do grid search over k, n_components and C:
	pipeline = Pipeline([("features", combined_features), ("svm", svm)])
	param_grid = dict(features__pca__n_components=[1, 2, 3],
			  features__univ_select__k=[1, 2],
			  svm__C=[0.1, 1, 10])
	grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
	grid_search.fit(Xtrain, trainlabel)
	print(grid_search.best_estimator_)

if "i" :
	mtest=int(fp.readline())

	cl=[]
	iden=[]
	inp=[]
	x=[]
	fvec=[]
	feature={}
	for _ in range(mtest):
	    x=fp.readline().strip().split(' ')
	    iden.append(x[0])
	    inp.append(x[1:])
	for entry in inp:
	    for i in range(n):
		key,val=entry[i].split(':')
		feature[key]=val
	    feature={int(key):float(val) for key,val in feature.items()}
	    fvec.append(feature.copy())
	for i in range(mtest):
	    row=[]
	    for key in sorted(fvec[i]):
		row.append(fvec[i][key])
	    row=np.matrix(row)
	    if i==0:
		test=row
	    else:
		test=np.r_[test,row]
	orig=['-1','+1']
	label2orig={i:orig[i] for i in range(2)}

	pred=cf.predict(test)
	cl=[0,1,1,0,1]
	for i,val in enumerate(cl):
	    print label2orig[val] 

	for i,val in enumerate(pred):
	    print iden[i],
	    print label2orig[val] 
