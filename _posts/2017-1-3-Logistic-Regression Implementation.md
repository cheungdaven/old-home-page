---
layout: post
comments: true
categories: MachineLearning
---

This code is from Machine Learning in Action.

{% highlight python %}
from math import *
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open("../dataset/testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    print(m,n)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha *  dataMatrix.transpose() * error # matrix multiplication
    return weights

def plotBestFit(wei):
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n= shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

def stocGradAscent(dataMatrix, classLabels, numIter=200):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = arange(m)
        print(dataIndex)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01 # alpha changes with each iteration
            randIndex = int(random.uniform(0, len(dataIndex)))
            print("randIndex="+str(randIndex))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            #del(dataIndex[randIndex])
    return weights

if __name__ == '__main__':
    dataArr,labelMat= loadDataSet()
    weights = stocGradAscent(array(dataArr),labelMat)
    print(weights)
    plotBestFit(weights)
{% endhighlight %} 

link of testSet: [testSet](https://raw.githubusercontent.com/pbharrin/machinelearninginaction/master/Ch05/testSet.txt)
