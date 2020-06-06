#robert maule prog 3
#4/20/20
import csv
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

def csv_to_list(path,  TargetCol=-1):
    outdat = []
    Target = []
    with open(path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if  TargetCol == -1:
                outdat.append(row[0:-1])
                Target.append(row[len(row)-1])
            else:
                outdat.append(row[0: TargetCol] + row[ TargetCol+1:])
                Target.append(row[ TargetCol])
        for j in range(len(outdat)):
            outdat[j] = [i for i in outdat[j]]
        Target = [float(i) for i in  Target]
        return outdat,  Target
    
    
def errList(data, weights,  Target):
    predictionlist = []
    for f, t in zip(data,  Target):
        pred, error, sqError = regList(f, weights, t)
        predictionlist.append([pred, error, sqError])
    return predictionlist


def regList(inFeature, inWeights,  Target):
    weightedlist = []
    [weightedlist.append(w*float(d)) for w,d in zip(inWeights, inFeature)]
    pred = sum(weightedlist)
    error = Target - pred
    sqError = error**2

    return pred, error, sqError


def errDelt(data, weights,  Target):
    deltalist = []
    for f,t in zip(data,  Target):
        pred, error, sqError = regList(f, weights, t)
        delt = []
        for d in f:
            delt.append(error*float(d))
        deltalist.append(delt)
    return deltalist

def sumErrorDelt(deltalist):
    return [sum(delt) for delt in zip(*deltalist)]

def newWeights(data, weights,  Target, alpha):
    deltalist = errDelt(data, weights,  Target)
    sumDelta = sumErrorDelt(deltalist)
    weights = [w + alpha * delt for w, delt in zip(weights, sumDelta)]
    return weights

def sumError(predictionlist):
    sumError = 0
    sqErr = 0
    for err in predictionlist:
        sumError += err[1]
        sqErr += err[2]
    sqErr /= 2
    return sumError, sqErr


def iterateWeights(n, data, weights,  Target, alpha):
    newW = weights
    for i in range(n):
        newW = newWeights(data, newW,  Target, alpha)
    return newW

d_raw,  Target = csv_to_list('prog3.txt')
data = []

for i in d_raw:
    row = i[1:-1]
    row.insert(0,1)
    data.append(row)

weights=[-0.146,0.185,-0.044,0.119]
alpha = 0.00000002
w1 = iterateWeights(1, data, weights,  Target, alpha)
w2 = iterateWeights(2, data, weights,  Target, alpha)
w100 = iterateWeights(100, data, weights,  Target, alpha)

squareErrroLi = []
xAxis = []
itterationI= 1
w = [-0.146,0.185,-0.044,0.119]

while itterationI <= 100:
    xAxis.append(itterationI)
    w = iterateWeights(1, data, w,  Target, alpha)
    squareErrroLi.append(sumError(errList(data, w,  Target))[1])
    itterationI += 1

plt.plot(xAxis, squareErrroLi)
plt.xlabel('iterations')
plt.ylabel('sum of squared errors')
plt.show()

print(squareErrroLi)
print("Initial weights:\n", weights)
print("\nPrediction     Error   Squared Error")
print(np.around(    np.asarray( errList(data, weights,  Target)), decimals=2   )   )
print("\nError Delta:")
print(np.around(    np.asarray( errDelt(data, weights,  Target)), decimals=2   )   )

print("\nWeights after first run ")
print(w1)
print("\nPrediction     Error   Squared Error")
print(np.around(    np.asarray( errList(data, w1,  Target)), decimals=2   )   )
print("\nError Delta:")
print(np.around(    np.asarray( errDelt(data, w1,  Target)), decimals=2   )   )

print("\nWeights after second run ")
print(w2)
print("\nPrediction     Error   Squared Error")
print(np.around(    np.asarray( errList(data, w2,  Target)), decimals=2   )   )
print("\nError Delta:")
print(np.around(    np.asarray( errDelt(data, w2,  Target)), decimals=2   )   )

print("\nNew weights after 100 runs ")
print(w100)
print("\nPrediction   Error  Squared Error")
print(np.around(    np.asarray( errList(data, w100,  Target)), decimals=2   )   )
print("\nError Delta:")
print(np.around(    np.asarray( errDelt(data, w100,  Target)), decimals=2   )   )
print("\nFinal Error: ", sumError(errList(data, w100,  Target))[1])
print("\n Part 2  \n")

d_raw,  Target = csv_to_list('prog3_2.txt',  TargetCol=1)
data = []

for i in d_raw:
    row = i[1:]
    row.insert(0,1)
    data.append(row)

weights=[-59.50,-0.15,0.60]
alpha = 0.000002
w1 = iterateWeights(1, data, weights,  Target, alpha)
w2 = iterateWeights(2, data, weights,  Target, alpha)
w100 = iterateWeights(100, data, weights,  Target, alpha)

squareErrroLi = []
xAxis = []
itterationI = 1
w = [-59.50,-0.15,0.60]

while itterationI <= 100:
    xAxis.append(itterationI)
    w = iterateWeights(1, data, w,  Target, alpha)
    squareErrroLi.append(sumError(errList(data, w,  Target))[1])
    itterationI += 1

plt.plot(xAxis, squareErrroLi)
plt.xlabel('iterations')
plt.ylabel('sum of squared errors')
plt.show()

print("Initial weights:\n", weights)
print("\nPrediction     Error   Squared Error")
print(np.around(    np.asarray( errList(data, weights,  Target)), decimals=2   )   )
print("\nError Delta:")
print(np.around(    np.asarray( errDelt(data, weights,  Target)), decimals=2   )   )

print("\nweights after one run ")
print(w1)
print("\nPrediction     Error   Squared Error")
print(np.around(    np.asarray( errList(data, w1,  Target)), decimals=2   )   )
print("\nError Delta:")
print(np.around(    np.asarray( errDelt(data, w1,  Target)), decimals=2   )   )

print("\n Weights after run 2")
print(w2)
print("\nPrediction     Error   Squared Error")
print(np.around(    np.asarray( errList(data, w2,  Target)), decimals=2   )   )
print("\nError Delta:")
print(np.around(    np.asarray( errDelt(data, w2,  Target)), decimals=2   )   )

print("\weight after 100 runs ")
print(w100)
print("\nPrediction   Error  Squared Error")
print(np.around(    np.asarray( errList(data, w100,  Target)), decimals=2   )   )
print("\nError Delta:")
print(np.around(    np.asarray( errDelt(data, w100,  Target)), decimals=2   )   )
print("\nFinal sum of squared errors: ", sumError(errList(data, w100,  Target))[1])
