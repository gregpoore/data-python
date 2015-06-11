def mode(arr) :
    f = {}
    arr=sorted(arr)
    count=0
    for a in arr : f[a] = f.get(a,0)+1
    m = max(f.values())
    for x in f:
       if f[x]==m:
          t = [(x,f[x])]
          count+=1
    if m>1:  
       return t[0][0] 
    else:
       return arr[0]
arr=[1,2,4,2,1,1,5,2,3,5]
print mode(arr)
