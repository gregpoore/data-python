from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
fp=open("trainingdata1.txt","rU")

m,n=map(int,fp.readline().split(' '))
cl=[];iden=[];
x=[];
for i in range(m):
    x=fp.readline().strip().split(' ')
    iden.append(x[0])
    cl.append(x[1])
    row=[]
    for item in x[2:]:
        val=float(item[(item.index(':')+1):])
        row.append(val)
    row=np.matrix(row)
    if i==0:
        train=row
    else:
        train=np.r_[train,row]

orig=['-1','+1']
orig2label={orig[i]:i for i in range(2)}
ytrain=[orig2label[i] for i in cl]

rf = RandomForestClassifier(n_estimators=10, criterion='gini',  min_samples_split=7, min_samples_leaf=4, max_features='auto')
clf = AdaBoostClassifier(base_estimator=rf, n_estimators=5, learning_rate=1.0, algorithm='SAMME.R', random_state=True)
clf.fit(train,ytrain)

mtest=int(fp.readline())
cl=[];iden=[];
x=[];
for i in range(mtest):
    x=fp.readline().strip().split(' ')
    iden.append(x[0])
    row=[]
    for item in x[1:]:
        val=float(item[(item.index(":")+1):])
        row.append(val)
    row=np.matrix(row)
    if i==0:
        test=row
    else:
        test=np.r_[test,row]

orig=['-1','+1']
label2orig={i:orig[i] for i in range(2)}

pred=clf.predict(test)

for i,val in enumerate(pred):
    print iden[i],
    print label2orig[val]

