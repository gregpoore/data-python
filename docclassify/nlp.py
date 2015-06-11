from random import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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

vectorizer=TfidfVectorizer(stop_words="english", max_df=0.5, sublinear_tf=True)
vectorizer.fit(traindata)
Xtrain=vectorizer.transform(traindata)

n_inp=int(raw_input())
results=[]
sentence=[]

for i in range(n_inp):
        sentence.append(raw_input().strip())
testdata=np.array(sentence)
Xtest=vectorizer.transform(testdata)

clf=MultinomialNB(alpha=0.1)
clf.fit(Xtrain,classdata)

pred=clf.predict(Xtest)
for i in pred:
    print i
print clf.score(Xtrain,classdata)
if "":
	for i in range(n_inp):
		print len(sentence[i]),sentence[i]
		word_class=[0]*len(classes)
		for word in sentence[i]:
		     for c,v in pb.iteritems():
			 if word in v:
			     word_class[int(c)-1]+=1
		print word_class     
