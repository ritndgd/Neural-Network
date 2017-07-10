import matplotlib.pylab as pylab
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style

xinit = 0.35
yinit = 0.45
eta = 0.01
xlist = []
ylist = []
energy = []

def calculateEnergy(x, y):
    res = (-(np.log10(1-x-y)))-np.log10(x)-np.log10(y)
    return res

def xval(x, y):
    resx = ((1 / (1 - x - y)) - 1 / x)
    return resx

def yval(x, y):
    resy = ((1 / (1 - x - y)) - 1 / y)
    return resy

for i in range(50):
    xtemp = xinit - (eta * (xval(xinit, yinit)))
    ytemp = yinit - (eta * (yval(xinit, yinit)))
    energy.append(calculateEnergy(xtemp, ytemp))
    xlist.append(xtemp)
    ylist.append(ytemp)
    xinit = xtemp
    yinit = ytemp

print(xlist)
print(ylist)

def plotTrajectory():
    z = []
    for i in range(0, 50):
        i += 1
        z.append(i)

    style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_wireframe(xlist,ylist,z)

    ax1.set_xlabel('x ->')
    ax1.set_ylabel('y ->')
    ax1.set_zlabel('z ->')

    plt.show()

def plotEnergy():
    pylab.plot(energy)
    pylab.show()

plotTrajectory()
plotEnergy()