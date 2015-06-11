from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import numpy as np
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
for _ in range(m):
    x=fp.readline().strip().split(' ')
#    print x
    iden.append(x[0])
    cl.append(x[1])
    inp.append(x[2:])
#    print iden
#    print cl
#    print inp
print "===================================================="
print "===================================================="
for entry in inp:
    for i in range(n):
        key,val=entry[i].split(':')
        feature[key]=val
    feature={int(key):float(val) for key,val in feature.items()}
    fvec.append(feature.copy())
print "===================================================="
#for entry in fvec:
#    print entry
print cl
#train=np.array([])
for i in range(m):
    #print fvec[i]
    row=[]
    for key in sorted(fvec[i]):
	row.append(fvec[i][key])
    row=np.matrix(row)
    if i==0:
        train=row
    else:
        train=np.r_[train,row]
	#train=np.r_(train,row)
print train 

orig=['-1','+1']
orig2label={orig[i]:i for i in range(2)}
#print orig2label
ytrain=[orig2label[i] for i in cl]
#print ytrain

model=LogisticRegression()
rfe=RFE(model,15)
rfe=rfe.fit(train,ytrain)

print(rfe.support_)
print(rfe.ranking_) #This is one of the results expected

#if "" :
print "===================================================="
print "===================================================="
print "===================================================="
print "====================ytest================================"
mtest=int(fp.readline())

print mtest
cl=[]
iden=[]
inp=[]
x=[]
fvec=[]
feature={}
for _ in range(mtest):
    x=fp.readline().strip().split(' ')
#    print x
    iden.append(x[0])
    inp.append(x[1:])
#    print iden
#    print cl
#    print inp
print "===================================================="
print "===================================================="
for entry in inp:
    for i in range(n):
	key,val=entry[i].split(':')
	feature[key]=val
    feature={int(key):float(val) for key,val in feature.items()}
    fvec.append(feature.copy())
for i in range(mtest):
    #print fvec[i]
    row=[]
    for key in sorted(fvec[i]):
	row.append(fvec[i][key])
    row=np.matrix(row)
    if i==0:
        test=row
    else:
        test=np.r_[test,row]
	#train=np.r_(train,row)
print test
orig=['-1','+1']
label2orig={i:orig[i] for i in range(2)}

pred=rfe.predict(test)
print pred
cl=[0,1,1,0,1]
for i,val in enumerate(cl):
    print label2orig[val] 

for i,val in enumerate(pred):
    print iden[i],
    print label2orig[val] 
