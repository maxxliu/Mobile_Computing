import numpy as np
import json
import localize
import sys

print("Mobile Computing | Lab 2 | Localization")
print("Andrea F. Daniele, Max X. Liu, Noah A. Hirsch")
print("")
print("Usage: python3.6 find_address.py [mac-address]")

areaWidth = 15          #how wide the area of possible mac locations is
graphResolution = 2     #how wide we want graph squares to be

address = sys.argv[1]
jsonData = localize.LoadData(address)
graph = localize.CreateGraph(areaWidth, graphResolution)
