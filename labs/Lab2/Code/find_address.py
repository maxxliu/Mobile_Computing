import numpy as np
import json
import localize
import sys
from utils import *

print "" 
print "Mobile Computing | Lab 2 | Localization"
print "Andrea F. Daniele, Max X. Liu, Noah A. Hirsch"
print ""
print "Usage: python find_address.py [mac-address]"

areaWidth = 15                                                              #how wide the area of possible mac locations is
graphResolution = 2                                                         #how wide we want graph squares to be
areaWidth = areaWidth + (graphResolution - (areaWidth % graphResolution))   #round width up to multiple of gR

#create graph and load data
graph = localize.CreateGraph(areaWidth, graphResolution)
address = sys.argv[1]
ID_to_MAC, MAC_to_ID, data = load_data('../Data/rssdataset')
mac_id = MAC_to_ID[address]


#for each data point, if it has the right MAC address, increment its intersections
for trace_id in data:
    trace_data = data[trace_id]
    useful_readings = [ trace_data[i,1] == mac_id for i in range(trace_data.shape[0]) ]

    for r in trace_data[ useful_readings ]:

        intersects = localize.ComputeIntersections(r[4], r[2], r[3], graphResolution, areaWidth / graphResolution)
        for i in intersects:
            graph[i[0], i[1]] += 1

#calculate which square holds the max
maxX = 0
maxY = 0
maxVote = 0
for i in range(areaWidth / graphResolution):
    for j in range(areaWidth / graphResolution):
        if graph[i, j] > maxVote:
            maxVote = graph[i, j]
            maxX = i
            maxY = j

print ""
print "Found Location:"
print " X: %d" % (maxX * graphResolution + graphResolution / 2)
print " Y: %d" % (maxY * graphResolution + graphResolution / 2)
