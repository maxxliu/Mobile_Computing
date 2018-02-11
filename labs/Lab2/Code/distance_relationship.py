import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

# These are the two devices that we know the location of:
# 8c:85:90:16:0a:a4
# location: x=6.8, y=6.8
# ac:9e:17:7d:31:e8
# location: x=-0.87, y=9.45

DEVICE_DATA = {'8c:85:90:16:0a:a4': {'x': 6.8, 'y': 6.8},
                'ac:9e:17:7d:31:e8': {'x': -0.87, 'y': 9.45}}

# All of the .txt files with RSS trace data
FILES = ['../Data/rssdataset/' + f for f in os.listdir('../Data/rssdataset')]

def get_distance():
    # Loop through each of the files and calucalate distance for the
    # known devices
    distance = []
    rss = []

    for f in FILES:
        with open(f, 'r') as data_file:
            json_data = data_file.read()
        data = json.loads(json_data)

        for d in data:
            mac = d['mac']
            if mac in DEVICE_DATA:
                dist = math.hypot(d['loc_x'] - DEVICE_DATA[mac]['x'],
                                        d['loc_y'] - DEVICE_DATA[mac]['y'])

            distance.append(dist)
            rss.append(int(d['rss']))

    distance = np.array(distance)
    rss = np.array(rss) * -1

    return distance, rss


def bucket_rss(distance, rss):
    '''
    create buckets for RSS and then put distances for same RSS
    into the buckets
    index of the list is the RSS
    value of index is a list of all the distances
    '''
    RSS_buckets = []
    for i in range(100):
        RSS_buckets.append([])

    for i, val in enumerate(rss):
        RSS_buckets[val].append(distance[i])

    return RSS_buckets


def calculate_buckets(RSS_buckets, b_size):
    '''
    RSS_buckets - index is the RSS value and value is a list of distances
    b_size - size of buckets to use when creating histogram
    '''
    # I want see which rss values have any distances associated with them
    rss_values = []
    for i, val in enumerate(RSS_buckets):
        if len(val) > 0:
            rss_values.append(i)

    # find the most common distance for each rss value
    distance = []
    for rss in rss_values:
        n, bins, patches = plt.hist(RSS_buckets[rss], b_size)
        # plt.show()
        # plt.close()
        largest = np.argmax(n)
        dist = (bins[largest] + bins[largest + 1]) / 2
        distance.append(dist)

    return rss_values, distance


def run_simple_distance_relationship():
    '''
    just goes through each rss and finds most common bucket for distance
    gives pretty bad results
    '''
    distance, rss = get_distance()
    RSS_buckets = bucket_rss(distance, rss)
    rss_values, dist = calculate_buckets(RSS_buckets, 50)

    # plot rss vs distance
    plt.close()
    plt.plot(rss_values, dist)
    plt.xlabel('RSS')
    plt.ylabel('Distance')
    plt.show()


def sliding_window():
    '''
    looks at a frame of data and throws away outliers
    '''


# # Create heatmap
# heatmap, xedges, yedges = np.histogram2d(RSS, DISTANCE, bins=(100,100))
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
# # Plot heatmap
# plt.clf()
# plt.title('RSS vs. DISTANCE')
# plt.ylabel('DISTANCE')
# plt.xlabel('RSS')
# plt.imshow(heatmap, extent=extent)
# plt.show()
# plt.plot(RSS, DISTANCE, 'ro')
# plt.xlabel('RSS')
# plt.ylabel('Distance')
# plt.show()
