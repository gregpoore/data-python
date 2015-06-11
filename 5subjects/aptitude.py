from random import random
import numpy as np
def iterate():
	n=int(fp.readline())
	#print n # no of students
	gpa=[]
	gpa=map(float,fp.readline().split(" "))
	#print gpa
        test=[]
        for i in range(5):
	    test.append(map(int,fp.readline().split(' ')))
        testarr=np.array(test)
	#print testarr
        gpac=[]
        for gpa in gpa:
            gpac.append(float(gpa*10))
        #print gpac
        diff=gpac-testarr 
        #print diff
        var=[]
        std=[]
        for entry in diff:
	    std.append(np.std(entry))
	    var.append(np.var(entry))
        #print var,std
        minv=np.Inf
        for i,v in enumerate(var):
            if v<minv:
               minv=v
               index=i
        print index+1
fp=open("file.txt","r")
x=int(fp.readline())
#print x
for _ in range(x):
        iterate()
