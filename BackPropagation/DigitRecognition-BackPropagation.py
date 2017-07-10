import numpy as np
import random
from struct import *
import matplotlib.pylab as mpl

epochVsMse = list()

def loadData(startN, N,bTrain):

    if bTrain:
        fImages = open('train-images.idx3-ubyte', 'rb')
        fLabels = open('train-labels.idx1-ubyte', 'rb')
    else:
        fImages = open('t10k-images.idx3-ubyte', 'rb')
        fLabels = open('t10k-labels.idx1-ubyte', 'rb')

    s1, s2, s3, s4 = fImages.read(4), fImages.read(4), fImages.read(4), fImages.read(4)
    mnIm = unpack('>I', s1)[0]
    numIm = unpack('>I', s2)[0]
    rowsIm = unpack('>I', s3)[0]
    colsIm = unpack('>I', s4)[0]
    fImages.seek(16 + startN * rowsIm * colsIm)

    mnL = unpack('>I', fLabels.read(4))[0]
    numL = unpack('>I', fLabels.read(4))[0]
    fLabels.seek(8 + startN)

    T = []

    for b in range(0, N):

        x = []
        for i in range(0, rowsIm * colsIm):
            val = unpack('>B', fImages.read(1))[0]
            x.append(val / 255.0)

        val = unpack('>B', fLabels.read(1))[0]
        y = []
        for i in range(0, 10):
            if val == i:
                y.append(1)
            else:
                y.append(0)
        T.append((x, y))

    fImages.close()
    fLabels.close()
    return T

class DigitRecognition():

    def __init__(self, inputNeurons, hiddenNeurons, outputNeurons):
        self.inputWeights = np.matrix([[random.uniform(-7, 7) for i in range(0,inputNeurons)] for j in range(0,hiddenNeurons)])
        self.hiddenWeights = np.matrix([[random.uniform(-7, 7) for i in range(0,hiddenNeurons)] for j in range(0,outputNeurons)])
        # self.bias =
        self.traingSet = loadData(0, 1000, True)
        self.eta = 0.003
        self.train(self.inputWeights, self.hiddenWeights)

    def tanHActivation(self, temp):
        return np.tanh(temp)

    def train(self, inputWeights, hiddenWeights):
        flag = True
        errors = list()
        global epochVsMse
        for run in range(1000):
            error = 0
            count = 0
            for i in range(len(self.traingSet)):
                xi = (self.traingSet[i][0])
                di = np.matrix(self.traingSet[i][1])
                xi = np.matrix(xi)
                ans = np.dot(xi, np.transpose(inputWeights))
                ansTemp = self.tanHActivation(ans)
                output = np.dot(ansTemp, np.transpose(hiddenWeights))
                outputFinal = self.tanHActivation(output)
                outputFinal = self.generateOutVector(np.argmax(outputFinal))
                outputFinal = np.matrix(outputFinal)

                temp = (np.multiply((np.dot((np.multiply((np.multiply(-2, np.subtract(di, outputFinal))), (np.subtract(1, np.power(np.tanh(output), 2))))), self.hiddenWeights)),(np.subtract(1, np.power(np.tanh(ans), 2)))))
                deltaInput = (np.dot((np.transpose(temp)), xi))
                inputWeights = np.subtract(inputWeights, np.multiply(self.eta, deltaInput))

                temp1 = (np.multiply(np.multiply(-2, np.subtract(di, outputFinal)), (np.subtract(1, np.power(np.tanh(output), 2)))))
                deltaHidden = (np.dot((np.transpose(temp1)), ansTemp))
                hiddenWeights = hiddenWeights - (self.eta * deltaHidden)

                error = error + np.linalg.norm(np.square(di - outputFinal))

            mse = error/len(self.traingSet)
            print(mse)
            # print(count)
            epochVsMse.append([run, mse])
            if(mse >= 0.1):
                flag = True
            else:
                flag = False
        self.plotEpochVsMse(epochVsMse)

    def generateOutVector(self, j):
        dXi = list()
        for i in range(0, 10):
            if i == j:
                dXi.append(1)
            else:
                dXi.append(0)
        return dXi

    def plotEpochVsMse(self, epochVsMse):
        xAxis = list()
        yAxis = list()

        for i in epochVsMse:
            xAxis.append(i[0])
            yAxis.append(i[1])
        mpl.plot(xAxis, yAxis, color='g')
        mpl.show()

DigitRecognition(784, 50, 10)