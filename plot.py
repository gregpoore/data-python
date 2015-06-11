import matplotlib.pyplot as plt
import numpy as np
f=np.genfromtxt(open("CAX_COPD_TRAIN_data.csv","rb"),delimiter=",",skip_header=1) #skiprows=1) 
#gaussian=[3,4,5,7,21,23,24,25,26,28,28,30,32,36,37,60]
for i in range(2,63):
#for i in gaussian :
   plt.figure(i)
   plt.hist(f[:,i])
   plt.title('Feature %d' % i)
   plt.show()

