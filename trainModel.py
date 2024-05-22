from util import *


trainingSetFilePath="uci-blood-transfusion-service-center\\transfusion.data.csv"
xTrain,yTrain=getData(trainingSetFilePath)
xTrain = xTrain.astype(float)
yTrain = yTrain.astype(float)

parameters(xTrain,yTrain)