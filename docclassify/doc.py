from random import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
fp=open("trainingdata.txt","r")
x=int(fp.readline())
print x
pb={}
classes=[]
for _ in range(x):
	p=[]
	p=fp.readline().strip().split(" ")
        if p[0] not in classes:
           classes.append(p[0])
	   pb[p[0]]=set(p[1:])
        else:
           for entry in p[1:]:
              pb[p[0]].add(entry)

if "":
	for key,value in pb.iteritems():
		print "==============================================="
		print key
		print len(value)

n_inp=int(raw_input())
results=[]
sentence=[]

for i in range(n_inp):
        sentence.append(raw_input().strip().split(" "))
print len(sentence)
if "":
	for i in range(n_inp):
		print len(sentence[i]),sentence[i]
		word_class=[0]*len(classes)
		for word in sentence[i]:
		     for c,v in pb.iteritems():
			 if word in v:
			     word_class[int(c)-1]+=1
		print word_class     
