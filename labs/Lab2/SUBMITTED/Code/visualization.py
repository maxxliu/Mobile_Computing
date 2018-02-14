# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Thursday, February 8th 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Tuesday, February 13th 2018


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from time import sleep
import math

class Viewer:

    def __init__(self, min_x=0, max_x=1, min_y=0, max_y=1, grid_res=0.1, autorescale=False):
        # axis limits
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.autorescale = autorescale
        self.grid_res = grid_res
        # create plot
        self.figure, self.ax = plt.subplots()
        # set axis lims (if needed)
        if autorescale:
            self.ax.set_autoscalex_on(True)
            self.ax.set_autoscaley_on(True)
        else:
            self.ax.set_xlim(self.min_x, self.max_x)
            self.ax.set_ylim(self.min_y, self.max_y)
        # create grid
        self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(grid_res))
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(grid_res))
        self.ax.grid(which='minor', alpha=0.7)
        # set aspect ratio
        self.ax.set_aspect('equal', 'datalim')
        # create list of 2d objects
        self.objects = []
        # open window
        plt.ion()
        plt.show()


    def update(self):
        # rescale (if needed)
        if self.autorescale:
            self.ax.relim()
            self.ax.autoscale_view()
        # draw and flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


    def create_path(self):
        path_id = len(self.objects)
        path, = self.ax.plot([],[])
        self.objects.append( path )
        return path_id


    def create_circle(self):
        circle_id = len(self.objects)
        circle = plt.Circle((0,0), radius=0, ec='red', fc='none')
        circle = self.ax.add_patch(circle)
        self.objects.append( circle )
        return circle_id

    def create_grid(self, gridmap_data, min_x, max_x, min_y, max_y):
        extent = (min_x, max_x, min_y, max_y)
        gridmap_id = len(self.objects)
        grid = plt.imshow(gridmap_data, cmap=plt.cm.Blues, interpolation='nearest', extent=extent, origin='lower')
        self.objects.append( grid )
        return gridmap_id

    def increment_path(self, path_id, trace, trace_pos):
        path = self.objects[path_id]
        path.set_xdata( trace[:trace_pos,2] )
        path.set_ydata( trace[:trace_pos,3] )

    def move_circle(self, circle_id, x, y, radius):
        circle = self.objects[circle_id]
        circle.center = (x, y)
        circle.set_radius( radius )

    def update_gridmap(self, gridmap_id, gridmap_data):
        grid = self.objects[gridmap_id]
        grid.set_data( gridmap_data )
        grid.autoscale()


    # def on_running(self, xdata, ydata):
    #     #Update data (with the new _and_ the old points)
    #     self.lines.set_xdata(xdata)
    #     self.lines.set_ydata(ydata)
    #     # rescale (if needed)
    #     if self.autorescale:
    #         self.ax.relim()
    #         self.ax.autoscale_view()
    #     #We need to draw *and* flush
    #     self.figure.canvas.draw()
    #     self.figure.canvas.flush_events()
    #
    #
    #
    #
    #
    # #Example
    # def __call__(self):
    #     import numpy as np
    #     import time
    #     self.on_launch()
    #     xdata = []
    #     ydata = []
    #     for x in np.arange(0,10,0.5):
    #         xdata.append(x)
    #         ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
    #         self.on_running(xdata, ydata)
    #         time.sleep(1)
    #     return xdata, ydata

    def show(self):
        plt.show()
