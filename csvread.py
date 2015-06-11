import csv
import numpy
f=open("CAX_COPD_TRAIN_data.csv","rb")
n=int(0)
p=int(0)
try:
    reader=csv.reader(f)
    reader=list(reader)
    result=numpy.array(reader[1:]).astype('float')
    X=reader[1:][2:]
    y=reader[1:][0]
    for entry in reader[1:]:
        if int(entry[1])==1:
            print entry[1],entry[2]
            p+=1
        elif int(entry[1])==0:
            print entry[1],entry[2]
            n+=1
    print p
    print n
    print n+p
    for res in y[:]:
        print res
    print reader.shape,reader.ndim
    print y.shape,y.ndim
finally:
    f.close()
