# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Tuesday, February 13th 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Wednesday, February 14th 2018


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


min_x = -30
max_x = 30
min_y = -10
max_y = 40
cell_width = 1.0
mac_to_show = "f8:cf:c5:97:e0:9e"




Z = np.load( "gridmap-%.2f-%s.npy" % ( cell_width, mac_to_show.replace(':', '_') ) )

x = np.arange(min_x, max_x, cell_width)
y = np.arange(min_y, max_y, cell_width)
X, Y = np.meshgrid(x,y)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

Z = Z[:,:60]
Z = Z / np.sum(Z)

print Z.shape

ax.plot_surface(X, Y, Z, cmap=plt.cm.Blues)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability')
ax.set_xlim(-30,30)

plt.show()
