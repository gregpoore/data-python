el=[]
el.append([])
el.append([])
el[0].append(1)
el[0].append(2)
el[0].append(3)
el[1].append(7)
el[1].append(8)
el[1].append(9)
XY=[]
for row in el:
  xy=1
  for i,col in enumerate(row):
    if (i==0) |(i==1):
      print i,col
      xy*=col
  XY.append(xy)
print el
print ''
print XY
