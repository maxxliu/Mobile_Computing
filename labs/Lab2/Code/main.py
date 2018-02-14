# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Thursday, February 8th 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Wednesday, February 14th 2018


from utils import *
import localization
import numpy as np
import sys

macs = {
    "8c:85:90:16:0a:a4" : (6.8, 6.8),
    "ac:9e:17:7d:31:e8" : (-0.87, 9.45),
    "d8:c4:6a:50:e3:b1" : None,
    "f8:cf:c5:97:e0:9e" : None
}

if len(sys.argv) != 3:
    print ""
    print "Mobile Computing | Lab 2 | Localization"
    print "Andrea F. Daniele, Max X. Liu, Noah A. Hirsch"
    print ""
    print "Usage: python main.py [mac-address] [resolution-in-meters]"
    exit()

mac_to_localize = sys.argv[1]
cell_width = float(sys.argv[2])

min_x = -30
max_x = 30  # 40 for f8:cf:c5:97:e0:9e only, others are fine with 30
min_y = -10
max_y = 40
gui = True                 # open a window showing the map in real-time
points_per_frame = 100      # update the windows once every points_per_frame points

# make sure that the gridmap can be fully discretized with the given cell_width
min_x = min_x - min_x % cell_width
max_x = max_x - max_x % cell_width
min_y = min_y - min_y % cell_width
max_y = max_y - max_y % cell_width

gridmap_x_width = max_x - min_x
gridmap_y_width = max_y - min_y

gridmap_i0 = int( abs(min_x) / cell_width )
gridmap_j0 = int( abs(min_y) / cell_width )

ID_to_MAC, MAC_to_ID, data = load_data('../Data/rssdataset')

graph = localization.create_graph( (gridmap_x_width,gridmap_y_width), cell_width )

if gui:
    from visualization import *
    v = Viewer( min_x, max_x, min_y, max_y, grid_res=cell_width )
    gridmap = v.create_grid( graph, min_x, max_x, min_y, max_y )
    current_measurement = v.create_circle()

# compute number of valid readings
num_valid_readings = 0
for trace_id in data:
    trace_data = data[trace_id]
    for i in range(trace_data.shape[0]):
        datapoint = trace_data[i]
        if datapoint[1] != MAC_to_ID[mac_to_localize]: continue
        num_valid_readings += 1

print 'Analyzing data: ',
pbar = ProgressBar( num_valid_readings )
for trace_id in data:
    trace_data = data[trace_id]
    if gui: path = v.create_path()
    for i in range(trace_data.shape[0]):
        datapoint = trace_data[i]
        if datapoint[1] != MAC_to_ID[mac_to_localize]: continue

        car_x = datapoint[2]
        car_y = datapoint[3]
        rss = datapoint[4]
        predicted_distance = localization.compute_distance(rss)

        intersections = localization.compute_intersections(car_x, car_y, rss, cell_width)

        for c in intersections:
            cell_pos = ( gridmap_j0 + c[1], gridmap_i0 + c[0] )
            graph[cell_pos] += 1

        # update the viewer
        if gui:
            v.increment_path( path, trace_data, i )
            v.move_circle( current_measurement, car_x, car_y, predicted_distance )
            v.update_gridmap( gridmap, graph )
            # update window if needed
            if i % points_per_frame == 0:
                v.update()
        # update progress bar
        pbar.next()

target_ji_location = np.unravel_index(np.argmax(graph), graph.shape) - np.asarray([gridmap_j0, gridmap_i0])
target_ij_location = target_ji_location[::-1]
target_xy_location = np.asarray(target_ij_location) * cell_width + cell_width / 2.0
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

# store gridmap for further analysis
gridmap_file = 'gridmap-%.2f-%s' % ( cell_width, mac_to_localize.replace(':', '_') )
np.save(gridmap_file, graph)
