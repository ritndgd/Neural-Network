import random
import matplotlib.pylab as mpl
import numpy as npy

class Perceptron():

    def __init__(self, N):
        self.X = self.generate_vectors(N)
        self.W = self.create_weight_list()
        self.TrainData = self.create_training_set()
        self.Miscalculations = self.perceptron_training_algorithm()
        self.plot()
        self.plot_epoch_misCal()

    def generate_vectors(self, N):
        X = list()
        for i in range(N):
            xi, xj = [random.uniform(-1, 1) for i in range(2)]
            X.append([1, xi, xj])
        return X

    def create_weight_list(self):
        W = list()
        w0 = random.uniform(-0.25, 0.25)
        w1 = random.uniform(-1, 1)
        w2 = random.uniform(-1, 1)
        W.append([w0, w1, w2])
        return W

    def create_training_set(self):
        TrainData = list()

        for i in self.X:
            ans = self.W[0][0] * i[0] + self.W[0][1] * i[1] + self.W[0][2] * i[2]
            if ans >= 0:
                TrainData.append([1.0, i[1], i[2]])
            else:
                TrainData.append([0.0, i[1], i[2]])
        return TrainData

    def perceptron_training_algorithm(self):
        w0, w1, w2 = [random.uniform(-1,1) for i in range(3)]
        print("New Weights")
        print(w0, w1, w2)
        w0t, w1t, w2t = w0, w1, w2
        epoch = 0
        eta = 1
        flag = True
        temp = 0
        count = 0
        misCalculations = list()
        while flag:
            count = 0
            for i in self.TrainData:
                ans = 1 * w0 + i[1] * w1 + i[2] * w2;
                if ans < 0:
                    temp = 0.0
                else:
                    temp = 1.0
                if temp != i[0]:
                    count += 1
                    w0 += (eta * 1) * (i[0] - temp)
                    w1 += (eta * i[1]) * (i[0] - temp)
                    w2 += (eta * i[2]) * (i[0] - temp)
                else:
                    continue
            misCalculations.append([epoch, count])
            if w0t == w0 and w1t == w1 and w2t == w2:
                flag = False
            else:
                w0t = w0
                w1t = w1
                w2t = w2
                flag = True;
                epoch += 1
        return misCalculations

    def plot(self):
        class0xaxis = list()
        class0yaxis = list()
        class1xaxis = list()
        class1yaxis = list()
        for i in self.TrainData:
            if i[0] == 1:
                class1xaxis.append(i[1])
                class1yaxis.append(i[2])
            else:
                continue
        mpl.scatter(class1xaxis, class1yaxis, color='c')

        for i in self.TrainData:
            if i[0] == 0:
                class0xaxis.append(i[1])
                class0yaxis.append(i[2])
            else:
                continue
        mpl.scatter(class0xaxis, class0yaxis, color='g')
        mpl.xlabel("x1")
        mpl.ylabel("x2")
        mpl.title("Training Set")
        mpl.text(1.5, 1, '0 class', style = 'italic', bbox={'facecolor': 'c', 'alpha': 0.3, 'pad': 5})
        mpl.text(1.5, 0.7, '1 class', style='italic', bbox={'facecolor': 'g', 'alpha': 0.3, 'pad': 5})
        x1 = npy.array(range(-1, 3))
        y1 = (self.W[0][0] + (self.W[0][1] * x1)) / -self.W[0][2]
        mpl.plot(x1, y1, color='k', label='Boundary')
        mpl.grid(True)
        mpl.show()

    def plot_epoch_misCal(self):
        xAxis = list()
        yAxis = list()

        for i in self.Miscalculations:
            xAxis.append(i[0])
            yAxis.append(i[1])
        mpl.scatter(xAxis, yAxis, color='g')
        mpl.xlabel("Miscalculations")
        mpl.ylabel("Epoch")
        mpl.title("Epoch Vs Miscalculations")
        x1 = npy.array(range(-1, 50))
        y1 = npy.array(range(-1, 50))
        mpl.grid(True)
        mpl.show()


Perceptron(1000)