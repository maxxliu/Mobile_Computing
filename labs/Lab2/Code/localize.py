import numpy as np
import json


def LoadData(address):

    rssJSON = []
    for i in range(1, 20):
        rssJSON.append(json.loads(open("../Data/rssdataset/rss-" + str(i) + ".txt").read()))

    addressData = []
    for i in rssJSON:
        for j in i:
            if j['mac'] == address:
                addressData.append(j)

    return addressData


def CreateGraph(width, squareWidth):

    #round the width up to a multiple of squareWidth
    width = width + (width - (width % squareWidth)

    #graph initialized to 0s
    #graph(0,0) represents the square at which:
    #   -bottom left corner is at x=0, y=0
    #   -top right corner is at x = squareWidth, y = squareWidth
    #first index is x-axis
    #second index is y axis
    graph = np.zeros(width / squareWidth, width / squareWidth)

    return graph
