from lcm_types.mcomp_msgs import experiment_log_msg
import pickle
import json
import os
import matplotlib.pyplot as plt

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

def plot_positions(files, test_type):
    '''
    plots locations of vehicles
    files (list) - list of files
    test_type (string) - '1', '2', '3', or '4'
    '''
    files = [f for f in files if 'broken' not in f]
    for f in files:
        file_path = DATA_DIRECTORY + test_type + "/" + f + "/data.dat"
        data = pickle.load(open(file_path,"r"))

        vehicle_positions = {}
        for msg in data:
            msg_timestamp, msg_data_encoded = msg
            msg_data = experiment_log_msg.decode(msg_data_encoded)

            veh = msg_data.vehicle
            loc = msg_data.vehicle_location
            if veh in vehicle_positions:
                vehicle_positions[veh].append(loc)
            else:
                vehicle_positions[veh] = [loc]

        # now we can plot the movement of the vehicles
        for key, value in vehicle_positions.items():
            x, y = locations_to_xy(value)
            plt.plot(x, y)

        plt.show()

def locations_to_xy(locations):
    '''
    takes list of coordinates and converts to a list of x values and a list
    of y values
    locations (list) - list of tuples that are coordinates
    '''
    x = []
    y = []
    for loc in locations:
        x.append(loc[0])
        y.append(loc[1])

    return x, y

plot_positions(ONE, '1')
