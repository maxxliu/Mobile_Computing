# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Saturday, February 10th 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Sunday, February 11th 2018


import numpy as np
import json
import math

def compute_distance(rss):
    return -rss * 0.05


# given car location, RSS measurement, and cell width, returns the list of cells (i.e., list
# of tuples) that are likely to be the true location of the target device
def compute_intersections(car_x, car_y, rss, cell_width, verbose=False):
    intersections = []
    cell_width = float(cell_width)
    car_location = np.asarray([car_x, car_y])
    # convert RSS into distance
    radius = float( compute_distance(rss) )
    # for efficiency, we don't need to look at the entire gridmap, but only at the smallest set of cells
    # containing a circle of radius `radius` (min_y and max_y are the same since the cells are squared)
    min_i = int( math.floor( (car_x - radius) / cell_width ) )
    max_i = int( math.ceil( (car_x + radius) / cell_width ) )
    min_j = int( math.floor( (car_y - radius) / cell_width ) )
    max_j = int( math.ceil( (car_y + radius) / cell_width ) )
    # print some verbose stuff
    if verbose:
        # compute number of cells to check
        num_cells_i = int( math.ceil( (max_i - min_i) / cell_width ) )
        num_cells_j = int( math.ceil( (max_j - min_j) / cell_width ) )
        print 'Car at (%.2f, %.2f), intersecting circle of radius %.2f, range [[%.2f, %.2f],[%.2f, %.2f]], [%d, %d] cells per side @ %.2f/cell' % (
            car_x, car_y, radius, min_i*cell_width, max_i*cell_width, min_j*cell_width, max_j*cell_width, num_cells_i, num_cells_j, cell_width
        )
    # check cells for intersection with circle of radius `radius`
    for i in range(min_i, max_i, 1):      # goes along X
        for j in range(min_j, max_j, 1):  # goes along Y
            # get the corners of this cells
            corners = get_corners(i, j, cell_width)
            # compute the number of corners within the circle
            distances = np.linalg.norm( corners - car_location, axis=1 )
            corners_inside = np.sum( distances < radius )
            # the circle intersects a cell only if some corners are inside and others outside the circle
            if corners_inside > 0 and corners_inside != 4:
                cell_pos = ( i, j )
                intersections.append( cell_pos )
    # return list of intersections
    return intersections


def create_graph(gridmap_shape, cell_width):
    # calculating the number of cells for each dimension
    num_x_cells = int(gridmap_shape[0] / cell_width)
    num_y_cells = int(gridmap_shape[1] / cell_width)
    # create empty gridmap
    graph = np.zeros( (num_y_cells,num_x_cells) )
    return graph


# returns the location of the corners of the cell (i,j).
# the results is a 2d array of shape 4x2: [bottom_left, bottom_right, top_right, top_left]
def get_corners(i, j, cell_width):
    corners = np.asarray([
        [i * cell_width, j * cell_width],
        [(i+1) * cell_width, j * cell_width],
        [(i+1) * cell_width, (j+1) * cell_width],
        [i * cell_width, (j+1) * cell_width]
    ])
    return corners
