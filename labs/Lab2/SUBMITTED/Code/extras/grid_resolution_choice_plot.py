# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Wednesday, February 14th 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Wednesday, February 14th 2018


import numpy as np
import matplotlib.pyplot as plt


N = 5
ind = np.arange(4)
mac1Err = [ 8.01, 3.98, 3.77, 4.20 ]
mac2Err = [ 2.65, 2.64, 2.37, 4.17 ]
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, mac1Err, width, align='center')
p2 = plt.bar(ind, mac2Err, width, bottom=mac1Err, color='orange', align='center')

plt.ylabel('Error (meters)')
plt.xlabel('Grid resolution (meters)')
plt.xticks(ind, ('0.2', '0.5', '1.0', '2.0'))
plt.legend((p1[0], p2[0]), ('8c:85:90:16:0a:a4', 'ac:9e:17:7d:31:e8'))

plt.show()
