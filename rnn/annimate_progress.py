import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import matplotlib.animation as animation
from time import sleep




fig = plt.figure()
#creating a subplot
ax1 = Axes3D(fig)


def animate(i):
    move = True
    while move:
        files = sorted([int(x.split("_")[1].split('.')[0]) for x in glob("./points_*.txt")])
        if len(files)>0:
            move = False
            f = files[-1]

    data = np.genfromtxt("points_{}.txt".format(f), delimiter=',')
    path = np.genfromtxt("paths_{}.txt".format(f), delimiter=',')
    for i,x in enumerate( path):
        if x[3]==-1:
            path[i][0] = x[0]/5
    path = path[:,[0,1,2,4,5,6]]
    color = [str(item/255.) for item in data[:,3]]

    ax1.clear()
    #ax1.scatter(data[:,1], data[:,0], data[:,2], s=data[:,3], color='brown')
    ax1.scatter(data[:,1], data[:,0], data[:,2], s=data[:,3]*2, alpha=0.1)
    for i in path:
        ax1.plot([i[0],i[3]], [i[1],i[4]], [i[2],i[5]], alpha=.4)

    ax1.text(1, 0, 0, str(f), color='red')
    ax1.set_xlabel("y")
    ax1.set_ylabel("x")
    plt.title('Live graph with matplotlib')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
