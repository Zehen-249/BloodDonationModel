from util import predict
import numpy as np
import csv


# w,b = parameters()
with open("param.csv",'r') as fileObj:
    reader=csv.reader(fileObj)
    next(reader)
    for row in reader:
        wTemp=row[0].strip('[')
        wTemp=wTemp.strip(']')
        wTemp=wTemp.split(' ')
        w=[]
        for i in wTemp:
            if i=="":
                continue
            w.append(float(i))
        
        w=np.array(w)
        b=float(row[1])
        
        


x=np.array(([2,15,3750,49],))

p=predict(x,w,b)

for i in p:
    if(i>=0.5):
        print("Will Donate with {}% Chances".format(i*100))
    else:
        print("Will Not Donate with {}% Chances".format(100-(i*100)))

