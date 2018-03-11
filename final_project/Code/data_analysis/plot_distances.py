from lcm_types.mcomp_msgs import experiment_log_msg
import pickle
import json
import os
import matplotlib.pyplot as plt


def calculate_av_distance(files, test_type):

    filecount = 0
    total = 0
    num_cars = 0

    print(test_type)

    files = [f for f in files if 'broken' not in f]
    for f in files:
        file_path = DATA_DIRECTORY + test_type + "/" + f + "/data.dat"

        try:
            data = pickle.load(open(file_path,"r"))
        except:
            continue

        print()
        print("NEW FILE")
        print()
        vehicles = {}
        for msg in data:
            msg_timestamp, msg_data_encoded = msg
            msg_data = experiment_log_msg.decode(msg_data_encoded)

            if msg_data.vehicle == '':
                continue

            if msg_data.vehicle not in vehicles.keys():
                vehicles[msg_data.vehicle] = 100000

            if ((msg_data.vehicle_location[0] - 2.68)**2 + (msg_data.vehicle_location[1] - 2.10)**2)**.5 < vehicles[msg_data.vehicle]:
                vehicles[msg_data.vehicle] = ((msg_data.vehicle_location[0] - 2.68)**2 + (msg_data.vehicle_location[1] - 2.10)**2)**.5

            print(vehicles)

        num_cars = len(vehicles)

        for key, value in vehicles.iteritems():
            total += value
        filecount += 1

    return total / (filecount * num_cars)

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

x = [1,2,3]
y = [calculate_av_distance(ONE, '1'), calculate_av_distance(TWO, '2'), calculate_av_distance(THREE, '3')]

#plt.plot(4, calculate_av_distance(FOUR, '4'))
plt.plot(x, y)
plt.show()
