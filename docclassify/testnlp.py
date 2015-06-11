from sklearn.cross_validation import train_test_split
from random import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
import matplotlib.pyplot as plt
fp=open("trainingdata.txt","rU")
x=int(fp.readline())
print x
traindata=[]
classdata=[]
for _ in range(x):
	p=fp.readline().strip()
	classdata.append(int(p[0]))
        traindata.append(p[1:].strip())
traindata=np.array(traindata)
classdata=np.array(classdata)
#for k in range(29,1,-1):
#	for i in range(randint(0,9),100,14):
train,cv,ytrain,ycv=train_test_split(traindata,classdata,test_size=0.33,random_state=14 ) #or pass Xand y as parameters to split func

vectorizer=TfidfVectorizer(stop_words="english", max_df=0.5, sublinear_tf=True)
vectorizer.fit(train)
Xtrain=vectorizer.transform(train)
Xcv=vectorizer.transform(cv)
max=0.0
i=0.0
for i in np.arange(0.001,0.5,0.001):
	clf=MultinomialNB(alpha=0.002)
#	clf=PassiveAggressiveClassifier(n_iter=9)
	clf.fit(Xtrain,ytrain)

	pred=clf.predict(Xcv)
	#for i in pred:
	#    print i
	#print clf.score(Xtrain,ytrain)
	print clf.score(Xcv,ycv)
        if max<clf.score(Xcv,ycv):
            max=clf.score(Xcv,ycv)
            index=i

print index,max
