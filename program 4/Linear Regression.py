#Robert Maule and Steve Soraino

import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import csv


def regList(inFeature, inWeights, intarget): # GOOD
    weighted = []
    [weighted.append(w*float(d)) for w,d in zip(inWeights, inFeature)]
    pred = sum(weighted)
    pred = 1 / (1 + math.exp(-pred))
    error = intarget - pred
    sqError = error**2

    return pred, error, sqError



def errDelt(data, weights, target):
    deltList = []
    for f,t in zip(data, target):
        pred, error, sqError = regList(f, weights, t)
        delt = []
        for d in f:
            delt.append((error * pred) * (1 - pred) * d)
        deltList.append(delt)
    return deltList

def csv_list(path, targetCol=-1):
    target = []
    outputdata = []
  
    with open(path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if targetCol == -1:
                outputdata.append(row[0:-1])
                target.append(row[len(row)-1])
            else:
                outputdata.append(row[0:targetCol] + row[targetCol+1:])
                target.append(row[targetCol])
        for j in range(len(outputdata)):
            outputdata[j] = [i for i in outputdata[j]]
        target = [float(i) for i in target]
        return outputdata, target




def sumerror(predList, div=2):
    sumError = 0
    squareError = 0
    for err in predList:
        sumError += err[1]
        squareError += err[2]
    squareError /= div
    return sumError, squareError


def errorlist(data, weights, target): # GOOD
    predList = []
    for f, t in zip(data, target):
        pred, error, sqError = regList(f, weights, t)
        predList.append([pred, error, sqError])
    return predList



def sumerrorDelt(deltList):
    return [sum(delta) for delta in zip(*deltList)]

def newWeights(data, weights, target, alpha):
    deltaList = errDelt(data, weights, target)
    sumDelt = sumerrorDelt(deltaList)
    weights = [w + alpha * delt for w, delt in zip(weights, sumDelt)]
    return weights



def plotReg(n, data, l_x, l_y, weights, target, alpha):
    XF = []
    YF = []
    XT = []
    YT = []
    for index, item in enumerate(data):
        if target[index] == 0:
            XF.append(item[1])
            YF.append(item[2])
        else:
            XT.append(item[1])
            YT.append(item[2])

    w = runweight(n, data, weights, target, alpha)
    y0 = -(w[0]/w[2]) - (w[1]/w[2]) * -1
    y1 = -(w[0]/w[2]) - (w[1]/w[2]) * 1

    plt.scatter(XF,YF,c='r')
    plt.scatter(XT,YT,c='b')
    plt.plot([-1,1],[y0,y1])
    line_leg = "v = " + str(-(w[0]/w[2]))[:4] + " + " + str((w[1]/w[2]))[:4] + "r"
    plt.legend([line_leg, "Bad", "Good"])
    plt.xlabel(l_x)
    plt.ylabel(l_y)
    plt.show()

def runweight(n, data, weights, target, alpha):
    newW = weights
    for i in range(n):
        newW = newWeights(data, newW, target, alpha)
    return newW
d_raw, target = csv_list('Table7_7.txt')
dataset = []
target_i = 0
for i in d_raw:
    dataset.append(i)
    dataset[target_i] = i[1:]
    target_i += 1
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(dataset)
dataset = scaler.transform(dataset)
dataset = np.insert(dataset, 0, 1, axis=1)
print(dataset[:5])
weights=[-2.9465,-1.0147,2.161]
alpha = 0.02
print("\nInitial weights:\n", weights)
pList = errorlist(dataset, weights, target)
dList = errDelt(dataset, weights, target)
print("\nErrors #1\n", np.asarray(pList[:5]))
print("\nErrors #1 deltas:\n", np.asarray(dList[:5]))
w1 = runweight(1, dataset, weights, target, alpha)
print("\nWeights after first run:\n", w1)
pList = errorlist(dataset, w1, target)
dList = errDelt(dataset, w1, target)
print("\Errors #2:\n", np.asarray(pList[:5]))
print("\nErrors #2 deltas:\n", np.asarray(dList[:5]))
w2 = runweight(1, dataset, w1, target, alpha)
print("\nweights after second run:\n", w2)
w2000 = runweight(2000, dataset, weights, target, alpha)
print("\nFinal weights after 2000 runs:\n", w2000)
pList = errorlist(dataset, w2000, target)
se, sqe = sumerror(pList, div=len(dataset))
print("\nFinal sum of squared errors: ", sqe)

sqerrorlist = []
xAxis = []
iterIndex = 1
w = weights.copy()

while iterIndex <= 2000:
    xAxis.append(iterIndex)
    w = runweight(1, dataset, w, target, alpha)
    sqerrorlist.append(sumerror(errorlist(dataset, w, target),div=len(dataset))[1])
    iterIndex += 1

plt.plot(xAxis, sqerrorlist)
plt.xlabel('runs')
plt.ylabel('sum of squares')
plt.show()

plotReg(1, dataset, 'RPM', 'Vibrations', weights, target, alpha)

plotReg(10, dataset, 'RPM', 'Vibrations', weights, target, alpha)

plotReg(200, dataset, 'RPM', 'Vibrations', weights, target, alpha)

plotReg(500, dataset, 'RPM', 'Vibrations', weights, target, alpha)

plotReg(2000, dataset, 'RPM', 'Vibrations', weights, target, alpha)
