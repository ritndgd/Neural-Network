import matplotlib.pylab as pylab
import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
def xval(x, y):
    resx = ((1 / (1 - x - y)) - 1 / x)
    return resx

def yval(x, y):
    resy = ((1 / (1 - x - y)) - 1 / y)
    return resy

def hmatrix(x, y):
    htemp = np.matrix([[math.pow((1 / (1 - x - y)), 2) + math.pow((1 / x), 2), math.pow((1 / (1 - x - y)), 2)],
                       [math.pow((1 / (1 - x - y)), 2), math.pow((1 / (1 - x - y)), 2) + math.pow((1 / y), 2)]])
    hinverse = inv(np.matrix(htemp))
    return hinverse

xinit = 0.75
yinit = 0.20

xlist = []
ylist = []
evalues = []
hmat = []
hinverse = []
energy = []

def calculateEnergy(x, y):
    res = (-(np.log10(1-x-y)))-np.log10(x)-np.log10(y)
    return res

for i in range(50):
    deltafx = [[xval(xinit, yinit)], [yval(xinit, yinit)]]
    hessian_gradient = np.dot(hmatrix(xinit, yinit), deltafx)
    xtemp = xinit - hessian_gradient.item(0)
    ytemp = yinit - hessian_gradient.item(1)
    energy.append(calculateEnergy(xtemp, ytemp))
    xlist.append(xtemp)
    ylist.append(ytemp)

    xinit = xtemp
    yinit = ytemp

pylab.plot(energy)
pylab.show()
# pylab.plot(xlist, ylist)
# pylab.show()
def plotTrajectory():
    z = []
    for i in range(0, 50):
        i += 1
        z.append(i)

    pylab.style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_wireframe(xlist,ylist,z)

    ax1.set_xlabel('x ->')
    ax1.set_ylabel('y ->')
    ax1.set_zlabel('z ->')

    plt.show()
plotTrajectory()
print(xlist)
print(ylist)