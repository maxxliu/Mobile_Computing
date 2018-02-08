import numpy as np
import json
import localize
import sys
from utils import *

print("Mobile Computing | Lab 2 | Localization")
print("Andrea F. Daniele, Max X. Liu, Noah A. Hirsch")
print("")
print("Usage: python find_address.py [mac-address]")

areaWidth = 15                                                              #how wide the area of possible mac locations is
graphResolution = 2                                                         #how wide we want graph squares to be
areaWidth = areaWidth + (graphResolution - (areaWidth % graphResolution))   #round width up to multiple of gR


graph = localize.CreateGraph(areaWidth, graphResolution)
address = sys.argv[1]
ID_to_MAC, MAC_to_ID, data = load_data('../Data/rssdataset')
