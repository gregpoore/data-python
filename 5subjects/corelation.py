from random import random
import numpy as np
def iterate():
	n=int(fp.readline())
	#print n # no of students
	gpa=[]
	gpa=map(float,fp.readline().split(" "))
	#print gpa
        testarr=[]
        for i in range(5):
	    testarr.append(map(float,fp.readline().split(' ')))
        testmean=[]
        teststd=[]
        for test in testarr: 
		teststd.append(np.std(test))
		testmean.append(np.mean(test))
        gpastd=np.std(gpa)
        gpamean=np.mean(gpa)
        print teststd,testmean 
        print gpastd,gpamean
        
        cor=[]
        for i,test in enumerate(testarr):
            xy=0
            for j in range(n):
                xy+=gpa[j]*test[j]
            num=(xy-n*testmean[i]*gpamean)
            #den=((n-1)*teststd[i]*gpastd)
            #cor.append(abs(num/den))
            try:
	       tmpcorr=float(num)/float(((n)*teststd[i]*gpastd))
	    except ZeroDivisionError:
	       if abs(teststd[i]-gpastd) < 1e-6:
	          tmpcorr = 1.0
	       else:
	          tmpcorr = 0.0
	    tmpcorr = abs(tmpcorr)
	    #print test,i,cor
            cor.append(tmpcorr)
        maxcor=0
        for i,c in enumerate(cor):
            print c
            if c>maxcor:
               maxcor=c
               index=i
        print index+1
fp=open("file.txt","r")
x=int(fp.readline())
#print x
for _ in range(x):
        iterate()
