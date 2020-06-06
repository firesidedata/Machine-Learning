#Robert MAule and Steve Soriano
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(suppress=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))

    return np.sum(first - second) / (len(X)) + reg


def gradient_with_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    #grad[0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    grad[0] = grad[0] - learningRate / len(X) * theta[0]

    return np.array(grad).ravel()


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x

    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax

def predict_oneinstance(X_inst, all_theta):
    num_labels = all_theta.shape[0]
    X_inst = np.insert(X_inst, 0, 1)
    X_inst = np.matrix(X_inst)
    all_theta = np.matrix(all_theta)

    # compute Probability
    h = sigmoid(X_inst * all_theta.T)
    print("\n Single Prediction not normalized")
    print(h)

    hlist = h[0].tolist()[0]
    h_norm = []

    for item in hlist:
        h_norm.append(item/sum(hlist))

    print("\n Single Prediction normalized")
    print(h_norm)
    print("\nSum of Normalized Predictions")
    print(sum(h_norm))

    # max prob
    h_argmax = np.argmax(h, axis=1)

    # true predicition
    h_argmax = h_argmax + 1
    print("\nClass Prediction:")
    print(h_argmax)
    return h_argmax

def csv_to_list(path, targCol=-1):
    outdat = []
    targ = []
    with open(path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if targCol == -1:
                outdat.append(row[0:-1])
                targ.append(row[len(row)-1])
            else:
                outdat.append(row[0:targCol] + row[targCol+1:])
                targ.append(row[targCol])
        for j in range(len(outdat)):
            outdat[j] = [i for i in outdat[j]]
        targ = [float(i) for i in targ]
        return outdat, targ

def plotReg(data, targ, thetas): #plotting data
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    for index, item in enumerate(data):
        if (targ[index] == 1):
            x1.append(item[0])
            y1.append(item[1])
        if (targ[index] == 2):
            x2.append(item[0])
            y2.append(item[1])
        if (targ[index] == 3):
            x3.append(item[0])
            y3.append(item[1])
    w1 = thetas[0]
    w2 = thetas[1]
    w3 = thetas[2]
    
    
    #line calculations for slope and intercepts
    
    yliney1 = -(w1[0]/w1[2]) - (w1[1]/w1[2]) * -1
    yliney2 = -(w1[0]/w1[2]) - (w1[1]/w1[2]) * 1
    line_leg1 = "f=" + str(-(w1[0]/w1[2]))[:5] + "+" + str(-(w1[1]/w1[2]))[:5] + "s"

    rliney1 = -(w2[0]/w2[2]) - (w2[1]/w2[2]) * -1
    rliney2 = -(w2[0]/w2[2]) - (w2[1]/w2[2]) * 1
    line_leg2 = "f=" + str(-(w2[0]/w2[2]))[:5] + "+" + str(-(w2[1]/w2[2]))[:5] + "s"

    bliney1 = -(w3[0]/w3[2]) - (w3[1]/w3[2]) * -1
    bliney2 = -(w3[0]/w3[2]) - (w3[1]/w3[2]) * 1
    line_leg3 = "f=" + str(-(w3[0]/w3[2]))[:5] + "+" + str(-(w3[1]/w3[2]))[:5] + "s"
#boudaries
    plt.ylim(bottom=-1.0, top=1.0)

#plot
    plt.scatter(x2, y2, c='r')
    plt.plot([-1,1],[rliney1 ,rliney2], c='r')

    plt.scatter(x1, y1, c='y')
    plt.plot([-1,1],[yliney1,yliney2], c='y')

    plt.scatter(x3, y3, c='b')
    plt.plot([-1,1],[ bliney1,bliney2], c='b')
    #plot legend
    plt.legend([line_leg1, line_leg2, line_leg3])
    plt.xlabel("SPEND")
    plt.ylabel("Frequency")
    plt.show()

def main():
    raw_data, target = csv_to_list('Table7_11.txt')
    for i in range(len(raw_data)):
        raw_data[i].append(target[i])
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(raw_data)
    raw_data = scaler.transform(raw_data)
    data = []
    for i in range(len(raw_data)):
        data.append(raw_data[i][:2].tolist())
    data = {'X': np.asarray(data).astype(np.float),
            'y': np.asarray(target).astype(np.float)}
        
    numLabels = 3
        
    
    params = data['X'].shape[1]

    all_theta = np.zeros((numLabels, params + 1))
    all_theta = one_vs_all(data['X'], data['y'], numLabels, 0.0001)
    print("Final Weights: ")
    print(all_theta)
    y_predict = predict_all(data['X'], all_theta)
    correct = [1 if a == b 
               else 0 
               for (a, b) 
               in zip(y_predict, data['y'])]
    accsum=sum(map(int, correct))
    accuracy = ( accsum / float(len(correct)))
    print("\nPredict Accuracy:")
    print('accuracy = {0}%'.format(accuracy * 100))
    plotReg(data['X'], target, all_theta)
    predict_oneinstance(np.array([0.10790978, 0.7643608]), all_theta)
    
main()