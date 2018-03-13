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
        type30_count = 0.0
        type30_byte_size = 0.0
        for msg in data:

            msg_timestamp, msg_data_encoded = msg
            msg_data = experiment_log_msg.decode(msg_data_encoded)

            # check message type really quickly
            if msg_data.type == 30:
                type30_count += 1
                type30_byte_size += len( bytes(msg_data) )

            veh = msg_data.vehicle
            loc = msg_data.vehicle_location
            if veh in vehicle_positions:
                vehicle_positions[veh].append(loc)
            else:
                vehicle_positions[veh] = [loc]

        # find how long the log was
        msg_timestamp, msg_data_encoded = data[0]
        msg_data = experiment_log_msg.decode(msg_data_encoded)
        begin_time = msg_data.timestamp
        msg_timestamp, msg_data_encoded = data[-1]
        msg_data = experiment_log_msg.decode(msg_data_encoded)
        end_time = msg_data.timestamp
        total_time = end_time - begin_time
        # i think total time is in milliseconds so I am going to change to seconds
        total_time = total_time / 1000
        # print the number of messages with type 30 for a specific file
        print ("For file %s there were %i messages with type 30 in %f seconds. That is %.2f messages/second, %.2f byte/sec" % (file_path, type30_count, total_time, type30_count/total_time, type30_byte_size/total_time))



        # commenting out the plotting code because i just want to see the messages/second

        # now we can plot the movement of the vehicles
        # for key, value in vehicle_positions.items():
        #     x, y = locations_to_xy(value)
        #     plt.plot(x, y)
        #
        # plt.show()

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
plot_positions(TWO, '2')
plot_positions(THREE, '3')
plot_positions(FOUR, '4')
