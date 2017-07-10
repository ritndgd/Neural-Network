import random
import numpy as np
import matplotlib.pylab as pylab

X = []
Y = []
x = 0
y = 0
avgx = 0
avgy = 0

for i in range(50):
    x += 1
    X.append(x)

print("X values")
print(X)
avgx = np.mean(X)

for j in range(50):
    y += 1
    rand = random.uniform(-1, 1)
    y += rand
    Y.append(y)

print("Y values")
print(Y)
avgy = np.mean(Y)

xdiffmean = []
ydiffmean = []

for k in range(50):
    temp = X[k] - avgx
    xdiffmean.append(temp)

for l in range(50):
    temp = Y[l] - avgy
    ydiffmean.append(temp)

ssxy = 0

for m in range(50):
    product = xdiffmean[m] * ydiffmean[m]
    ssxy = ssxy + product

ssxx = 0

for n in range(50):
    xsq = xdiffmean[n] * xdiffmean[n]
    ssxx = ssxx + xsq

w1 = ssxy / ssxx

w0 = avgy - (w1 * avgx)

print("Weights to minimize function:")
print(w0, w1)

X1 = []
Y1 = []

X1 = np.array(range(-3, 55))

Y1 = (w0 + (w1 * X1))

pylab.scatter(X, Y, color='b')
pylab.plot(X1, Y1, color='r', label='Boundary')
pylab.xlabel("x ->")
pylab.ylabel("y ->")
pylab.grid(True)
pylab.show()