# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Tuesday, February 13th 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Wednesday, February 14th 2018


import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        RSS_buckets[int(val)].append(distance[i])

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
            # print "rss value: %i number of observations: %i" % (i, len(val))
            rss_values.append(i)

    # find the most common distance for each rss value
    distance = []
    for rss in rss_values:
        n, bins, patches = plt.hist(RSS_buckets[rss], b_size)
        # plt.show()
        # plt.close()
        largest = np.argmax(n)
        dist = (bins[largest] + bins[largest + 1]) / 2

        # what if i did median instead
        # i tried this and it looks very noisy
        # i = len(RSS_buckets[rss]) // 2
        # dist = RSS_buckets[rss][i]

        # ok what if i did mean
        # the curve is almost linear but looks pretty smooth
        # could definitely try this and see if it gets better results
        # dist = np.mean(RSS_buckets[rss])

        distance.append(dist)

    return rss_values, distance


def run_simple_distance_relationship():
    '''
    just goes through each rss and finds most common bucket for distance
    '''
    distance, rss = get_distance()
    RSS_buckets = bucket_rss(distance, rss)
    rss_values, dist = calculate_buckets(RSS_buckets, 20)

    # plot rss vs distance
    plt.close()
    plt.plot(rss_values, dist)
    plt.xlabel('RSS')
    plt.ylabel('Distance')
    plt.show()


def moving_average(a, size):
    '''
    a - np array
    size - size of window
    '''
    ret = np.cumsum(a)
    ret[size:] = ret[size:] - ret[:-size]
    return ret[size - 1:] / size


def load_data_ma(wsize=4):
    '''
    loads data but takes moving average first
    '''
    # Loop through each of the files and calucalate distance for the
    # known devices
    distance = np.array([])
    rss = np.array([])

    for f in FILES:
        with open(f, 'r') as data_file:
            json_data = data_file.read()
        data = json.loads(json_data)

        dist_temp = []
        rss_temp = []
        for d in data:
            mac = d['mac']
            if mac in DEVICE_DATA:
                dist = math.hypot(d['loc_x'] - DEVICE_DATA[mac]['x'],
                                        d['loc_y'] - DEVICE_DATA[mac]['y'])

                dist_temp.append(dist)
                rss_temp.append(int(d['rss']))

        dist_temp = moving_average(np.array(dist_temp), wsize)
        rss_temp = np.array(rss_temp) * -1
        rss_temp = moving_average(np.array(rss_temp), wsize)

        distance = np.hstack([distance, dist_temp])
        rss = np.hstack([rss, rss_temp])

    return distance, rss


def run_moving_avg_distance_relationship(bsize=20, wsize=4):
    '''
    create relationship using moving average
    '''
    distance, rss = load_data_ma(wsize)
    RSS_buckets = bucket_rss(distance, rss)
    rss_values, dist = calculate_buckets(RSS_buckets, bsize)

    # only want to use the range from [39, 64]
    rss_values = np.array(rss_values[:-6])
    dist = np.array(dist[:-6])

    # return rss_values, dist

    # find best fit line
    z = np.polyfit(rss_values, dist, 2)
    p2 = np.poly1d(z)

    xp = np.linspace(30, 70, 200)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plot rss vs distance
    # plt.close()
    ax.plot(rss_values, dist, xp, p2(xp), '-')
    ax.set_xlabel('RSS (dBm)')
    ax.set_ylabel('Distance (meters)')
    plt.xlim(39, 64)
    plt.ylim(0, 12)

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("-%d"))

    plt.show()

    # return the best fit curve
    return p2

run_moving_avg_distance_relationship()

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
