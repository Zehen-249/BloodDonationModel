import csv
import numpy as np


def getData(filePath):
    with open(filePath,'r') as file:
        data=csv.reader(file)
        next(data)

        xTrain=[]
        yTrain=[]

        lenData=0
        for row in data:
            xTrain.append(row[:4])
            yTrain.append(row[4])
            lenData+=1

        xTrain=np.array(xTrain)
        yTrain=np.array(yTrain)
        
        return xTrain,yTrain
