import numpy as np
import json

def ComputeDistance(strength):

    return (-strength) / 10


#for a given RSS strength, car location, and graph, compute squares to increment
def ComputeIntersections(strength, x, y, width, diagonal, dimension):

    outerRadius = ComputeDistance(strength)
    innerRadius = outerRadius - diagonal
    outerIntersection = False
    innerIntersection = False
    intersections = []

    for i in range(dimension):                 #x direction
        for j in range(dimension):             #y direction
            outerIntersection = False
            innerIntersection = False

            xDistance = abs(x - (i * width + (width / 2)))
            yDistance = abs(y - (j * width + (width / 2)))

            if xDistance > ((width / 2) + outerRadius):
                continue
            if yDistance > ((width / 2) + outerRadius):
                continue

            if xDistance <= width / 2:
                outerIntersection = True
                innerIntersection = True
            if yDistance <= width / 2:
                outerIntersection = True
                innerIntersection = True

            cornerDistance = (xDistance - (width/2))**2 + (yDistance - (width/2))**2
            if outerIntersection != True:
                if cornerDistance <= outerRadius**2:
                    outerIntersection = True


            if xDistance > ((width / 2) + innerRadius):
                innerIntersection = False
            if yDistance > ((width / 2) + innerRadius):
                innerIntersection = False

            if innerIntersection != True:
                if cornerDistance <= innerRadius**2:
                    innerIntersection = True

            if outerIntersection and not innerIntersection:
                intersections.append([x, y])

    return intersections



#creates a graph for localization
def CreateGraph(width, squareWidth):

    #calculating size of each dimension
    w = int(width / squareWidth)

    #graph initialized to 0s
    #graph(0,0) represents the square at which:
    #   -bottom left corner is at x=0, y=0
    #   -top right corner is at x = squareWidth, y = squareWidth
    #first index is x-axis
    #second index is y axis
    graph = np.zeros((w,w))

    return graph


#returns locations of corners of given square within graph
#begins at bottom left corner (min x & min y), and rotates counter clockwise
def ComputeSquareLocation(x, y, width):

    dimensions = [[x * width, y * width],
                  [(x + 1) * width, y * width],
                  [(x + 1) * width, (y + 1) * width],
                  [x * width, (y + 1) * width]]
    return dimensions
