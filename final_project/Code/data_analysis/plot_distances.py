from lcm_types.mcomp_msgs import experiment_log_msg
import pickle
import json
import os
import matplotlib.pyplot as plt
import numpy as np

verbose = False

perception_distance = 0.20
scale_factor = 20

POI = np.asarray([2.68, 2.10])

def dAB( A, B ):
    A = np.asarray(A)
    B = np.asarray(B)
    return np.linalg.norm(A-B)


def calculate_distances(files, test_type, offset=0.0):

    filecount = 0
    # total = 0
    num_cars = 0

    distances = []
    inter_vehicle_distances = []

    if verbose:
        print(test_type)

    files = [f for f in files if 'broken' not in f]
    for f in files:
        file_path = DATA_DIRECTORY + str(test_type) + "/" + f + "/data.dat"

        try:
            data = pickle.load(open(file_path,"r"))
        except:
            continue

        if verbose:
            print()
            print("NEW FILE")
            print()
        vehicles = {}
        for msg in data:
            msg_timestamp, msg_data_encoded = msg
            msg_data = experiment_log_msg.decode(msg_data_encoded)

            if msg_data.vehicle == '':
                continue


            # if test_type == 1 and msg_data.vehicle_location[1] > POI[1]:
            #     continue

            if msg_data.vehicle not in vehicles.keys():
                vehicles[msg_data.vehicle] = 999999



            d = dAB( msg_data.vehicle_location, POI )

            if d < vehicles[msg_data.vehicle]:
                vehicles[msg_data.vehicle] = d

            if verbose:
                print(vehicles)

        num_cars = len(vehicles)

        last_distance = None

        print sorted(vehicles.values())
        tmp = []

        for value in sorted(vehicles.values()):
            distance = value - offset
            distances.append( distance )

            if last_distance is not None:
                inter_vehicle_distances.append( distance - last_distance )

                tmp.append( distance - last_distance )

            last_distance = distance

        filecount += 1

        print tmp

    # return total / (filecount * num_cars)
    return np.asarray(distances)*scale_factor, np.asarray(inter_vehicle_distances)*scale_factor

# get all of the file names
DATA_DIRECTORY = "../../Data/"
for experiment in os.listdir(DATA_DIRECTORY):
    tmp = DATA_DIRECTORY + experiment
    if experiment == '1':
        ONE = os.listdir(tmp) # iles for nothing
    elif experiment == '2':
        TWO = os.listdir(tmp) # files for communication, no propogation
    elif experiment == '3':
        THREE = os.listdir(tmp) # files for communication, yes propogation
    elif experiment == '4':
        FOUR = os.listdir(tmp) # files for communication, propogation, broadcast

distances = [ 0.0 ] * 4
inter_veh_distances = [ 0.0 ] * 4

ex_id_to_files = {
    1 : ONE,
    2 : TWO,
    3 : THREE,
    4 : FOUR
}

for i in range(4):
    distances[i], inter_veh_distances[i] = calculate_distances(ex_id_to_files[i+1], i+1, perception_distance)


# distances = [
#     calculate_distances(ONE, '1', perception_distance),
#     calculate_distances(TWO, '2', perception_distance),
#     calculate_distances(THREE, '3', perception_distance),
#     calculate_distances(FOUR, '4', perception_distance)
# ]

x = [1,2,3,4]
y = [
    np.average( distances[i] )
    for i in range(len(distances))
]

std = [
    np.std( distances[i] )
    for i in range(len(distances))
]


y_inter = [
    np.average( inter_veh_distances[i] )
    for i in range(len(inter_veh_distances))
]

std_inter = [
    np.std( inter_veh_distances[i] )
    for i in range(len(inter_veh_distances))
]


# y = [calculate_av_distance(ONE, '1'), calculate_av_distance(TWO, '2'), calculate_av_distance(THREE, '3'), calculate_av_distance(FOUR, '4')]

# std_dev = [
#
#     for i in range(len(y))
# ]

# y = [ y[i]-y[0]+.03 for i in range(len(y)) ]


for i in range( min(len(x), len(y)) ):
    print 'Experiment #%d: avg distance from POI: %.2f meters +/- %.2f; avg inter-vehicle distance: %.2f meters +/- %.2f; (scaled w/ factor= 1:%d)' % ( x[i], y[i], std[i], y_inter[i], std_inter[i], scale_factor )

#plt.plot(4, calculate_av_distance(FOUR, '4'))
plt.plot(x, y)
# plt.show()
