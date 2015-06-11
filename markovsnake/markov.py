from random import random
import numpy as np
def iterate():
	p=[]
	p=map(float,fp.readline().split(","))
	print p
	nlad,nsnake=map(int,fp.readline().split(','))
	print nlad,nsnake
	ladder={}
	snakes={}
	x=fp.readline().strip().split(" ")
	for x in x:
	   key,value=map(int,x.split(","))
	   ladder[key]=value
	print ladder
	x=fp.readline().strip().split(" ")
	for x in x:
	   key,value=map(int,x.split(","))
	   snakes[key]=value
	print snakes
	dist=np.array([p[0],p[0]+p[1],p[0]+p[1]+p[2],p[0]+p[1]+p[2]+p[3],p[0]+p[1]+p[2]+p[3]+p[4],p[0]+p[1]+p[2]+p[3]+p[4]+p[5]])

	print dist
	repeat=0
        repeatsum=0
        repeatcount=50
        while(repeat<repeatcount):
		outcome=int(0)
		val=int(0)
		count=0
		while((val!=100) & (count <1000)):
			f=float(random())
			if (f>=0.0) & (f<dist[0]):
			   outcome=1
			elif (f>=dist[0]) & (f<dist[1]):
			   outcome=2
			elif (f>=dist[1]) & (f<dist[2]):
			   outcome=3
			elif (f>=dist[2]) & (f<dist[3]):
			   outcome=4
			elif (f>=dist[3]) & (f<dist[4]):
			   outcome=5
			elif (f>=dist[4]) & (f<=dist[5]):
			   outcome=6
			
			if (val+outcome) in range(101):
			   val+=outcome
			#print val
			count+=1
			for key,value in ladder.iteritems():
			   if val==key:
			      val=value
			      break
			for key,value in snakes.iteritems(): 
			   if val==key:
			      val=value
			      break
		   
		#print count
                repeat+=1
                repeatsum+=count
	print repeatsum/repeatcount
fp=open("file.txt","r")
x=int(fp.readline())
print x
for _ in range(x):
        iterate()
