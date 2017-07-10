import random
import math
import matplotlib.pyplot as mpl

finalOutput = list()
epochVsMse = list()

class FeedForward_Backpropagate():

    def __init__(self, N, M):
        self.X = self.generateInputs(N)
        self.Wih = self.generateWeights(M)
        print("Initial Wih", format(self.Wih))
        self.Who = self.generateWeights(M)
        print("Initial Who", format(self.Who))
        self.V = self.generateV(N);
        self.D = self.generateDesiredOutput(N, self.V, self.X)
        print("Desired", format(self.D))
        self.Bias = self.generateWeights(M)
        print("Bias", format(self.Bias))
        self.train(self.X, self.Wih, self.Who, self.Bias, self.D)

    def generateInputs(self, N):
        X = list()
        for i in range(N):
            xi = random.uniform(0,1)
            X.append(xi)
        return X

    def generateV(self, N):
        V = list()
        for i in range(N):
            vi = random.uniform(-1/10, 1/10)
            V.append(vi)
        return V

    def generateDesiredOutput(self, N, V, X):
        D = list()
        for i in range(N):
            di = math.sin(20*X[i]) + 3 * X[i] + V[i]
            D.append(di)
        return D

    def generateWeights(self, M):
        W = list()
        for i in range(M):
            wi = random.uniform(-4, 4)
            W.append(wi)
        return W

    def train(self, X, W, Who, Bias, D):
        previousMse = 0
        epoch = 1
        global epochVsMse
        eta = 0.01
        global finalOutput
        biasOfOutput = random.uniform(-4, 4)
        flag = True
        while(flag):
            Result = list()
            for i in range(len(X)):
                xi = X[i]
                tempList = list()
                tempNonTan = list()
                aj = list()
                for j in range(len(W)):
                    temp = (xi * W[j]) + Bias[j]
                    tempNonTan.append(temp)
                    temp = self.tanHActivation(temp)
                    aj.append(temp);
                    temp = temp * Who[j]
                    tempList.append(temp)
                a = sum(tempList) + biasOfOutput
                Result.append(a)
                for n in range(24):
                    deltaj = (-2 * (D[i] - a)) * ((1-math.pow(math.tanh(tempNonTan[n]), 2)) * xi * Who[n])
                    W[n] = W[n] - (eta * deltaj)

                for m in range(24):
                    deltaBias = (-2 * (D[i] - a)) * ((1-math.pow(math.tanh(tempNonTan[m]), 2)) * Who[m])
                    Bias[m] = Bias[m] - (eta * deltaBias)

                for k in range(24):
                    deltai = ((-2 * (D[i] - a)) * aj[k])
                    Who[k] = Who[k] - (eta * deltai)
                biasOfOutput = biasOfOutput - (eta * (-2 * (D[i] - a)))
            epoch = epoch + 1
            mse = 0;
            for i in range(N):
                mse = mse + (math.pow((D[i] - Result[i]),2))
            mse = mse / N
            epochVsMse.append([epoch, mse])
            print(mse)
            finalOutput = Result
            if (mse >= 0.005):
                flag = True
            else:
                flag = False
        print("Epoch = ", format(epoch))
        print("Final Wih", format(W))
        print("Final Wih", format(W))
        print("Final Bias", format(Bias))
        print("Final Output Bias", biasOfOutput)
        print("Final Output", finalOutput)
        mpl.xlabel("Input")
        mpl.ylabel("Output")
        mpl.text(0, 3.5, '(x, d); x:0,1', style='italic', bbox={'facecolor': 'b', 'alpha': 0.3, 'pad': 5})
        mpl.scatter(X, D, color='b')
        mpl.show()
        mpl.text(0, 3.5, '(x, d); x:0,1', style = 'italic', bbox={'facecolor': 'b', 'alpha': 0.3, 'pad': 5})
        mpl.text(0, 3, '(x, f(x,w0)); x:0,1', style='italic', bbox={'facecolor': 'r', 'alpha': 0.3, 'pad': 5})
        mpl.scatter(X, D, color = 'b')
        mpl.scatter(X, finalOutput, color = 'r')
        mpl.show()
        self.plotEpochVsMse(epochVsMse)

    def tanHActivation(self, temp):
        return math.tanh(temp)

    def plotGrapgh(self, X, D) :
        mpl.xlabel("Input")
        mpl.ylabel("Output")
        mpl.plot(X,D)
        mpl.show()

    def plotEpochVsMse(self, epochVsMse):
        xAxis = list()
        yAxis = list()

        for i in epochVsMse:
            xAxis.append(i[0])
            yAxis.append(i[1])
        mpl.xlabel("Epoch")
        mpl.ylabel("MSE")
        mpl.title("Epoch Vs MSE")
        mpl.plot(xAxis, yAxis, color='g')
        mpl.show()

N = 300
M = 24
ff = FeedForward_Backpropagate(N, M)
