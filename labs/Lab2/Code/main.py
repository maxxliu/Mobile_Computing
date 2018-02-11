# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Thursday, February 8th 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Sunday, February 11th 2018


from utils import *
from visualization import *
import localization
import numpy as np


macs = {
    "8c:85:90:16:0a:a4" : (6.8, 6.8),
    "ac:9e:17:7d:31:e8" : (-0.87, 9.45),
    "d8:c4:6a:50:e3:b1" : None,
    "f8:cf:c5:97:e0:9e" : None
}

mac_to_localize = "ac:9e:17:7d:31:e8"

min_x = -6
max_x = 12
min_y = -2
max_y = 12
cell_width = .25            # in meters
points_per_frame = 100      # update the windows once every points_per_frame points

# make sure that the gridmap can be fully discretized with the given cell_width
min_x = min_x - min_x % cell_width
max_x = max_x - max_x % cell_width
min_y = min_y - min_y % cell_width
max_y = max_y - max_y % cell_width

gridmap_x_width = max_x - min_x
gridmap_y_width = max_y - min_y

gridmap_i0 = int( -min_x / cell_width )
gridmap_j0 = int( -min_y / cell_width )

ID_to_MAC, MAC_to_ID, data = load_data('../Data/rssdataset')

graph = localization.create_graph( (gridmap_x_width,gridmap_y_width), cell_width )

v = Viewer( min_x, max_x, min_y, max_y, grid_res=cell_width )

gridmap = v.create_grid( graph, min_x, max_x, min_y, max_y )

for trace_id in data:
    trace_data = data[trace_id]
    path = v.create_path()
    current_measurement = v.create_circle()
    for i in range(trace_data.shape[0]):
        datapoint = trace_data[i]
        if datapoint[1] != MAC_to_ID[mac_to_localize]: continue

        v.increment_path( path, trace_data, i )

        car_x = datapoint[2]
        car_y = datapoint[3]
        rss = datapoint[4]
        predicted_distance = localization.compute_distance(rss) # TODO: update

        v.move_circle( current_measurement, car_x, car_y, predicted_distance )

        intersections = localization.compute_intersections(car_x, car_y, rss, cell_width)

        for c in intersections:
            cell_pos = ( gridmap_j0 + c[1], gridmap_i0 + c[0] )
            graph[cell_pos] += 1

        v.update_gridmap( gridmap, graph )

        if i % points_per_frame == 0:
            v.update()

target_ji_location = np.unravel_index(np.argmax(graph), graph.shape)
target_ij_location = target_ji_location[::-1]
target_xy_location = np.asarray(target_ij_location) * cell_width
print 'Target device "%s" located at i,j: %r,  x,y: %r' % (
    mac_to_localize,
    target_ij_location,
    target_xy_location
)

# compute error (if possible)
if macs[mac_to_localize] is not None:
    gtruth = np.asarray( macs[mac_to_localize] )
    prediction_error = np.linalg.norm( gtruth-target_xy_location )
    print "Prediction error: %.2f meters" % prediction_error
