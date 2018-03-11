#Lab3

from mypicar.front_wheels import Front_Wheels
from mypicar.back_wheels import Back_Wheels
from detector_wrapper import DetectorWrapper
import time
import numpy as np

# create line detector
detector = DetectorWrapper()

# initialize interface to front and back wheels
front_wheels = Front_Wheels()
back_wheels = Back_Wheels()

# set parameters
max_speed = 20
turn_gain = .25
turn_offset = -2
x_offset = 0
use_memory = False
memory = []
memory_size = 3

# stop the car and set the front wheels straight
back_wheels.stop()
front_wheels.turn_rel( turn_offset )

# the adventure begins
i = 0
try:
    while True:
        # get new frame from the camera and detect lines
        success, ret = detector.detect()
        detector.plot(ret)

        # if the lines were successfully detected
        if success:
            i += 1

            # set speed and direction for back_wheels
            back_wheels.speed = max_speed
            back_wheels.forward()

            # get info about the lines
            frame, mid_x, left_fit, right_fit, ploty, left_fitx, right_fitx = ret

            # get ideal position for mid point (we found 300 to work better than img_width/2)
            img_width = frame.shape[0]
            ideal_mid_x = 300

            # compute error
            error = mid_x - ideal_mid_x

            # get a turning angle that is proportional to the error
            turn_angle = turn_offset + error * turn_gain

            # skip the first 8 frames from the camera due to lighting artifact that allucinate phantom lines
            if i < 8: continue

            # store action in memory
            memory.append( turn_angle )

            # compute average angle (in case of memory-based auto-pilot)
            if use_memory:
                turn_angle = 0.0
                mem_size = min(len(memory),memory_size)
                for i in range(1,mem_size,1):
                    turn_angle += ( float(mem_size-i+1)/float(mem_size) ) * memory[-i]

            # some debugging info
            print 'mid-point @ %d pxs / %d pxs, turning @ %.2f' % (mid_x, img_width, turn_angle)

            # turn to correct the heading error
            front_wheels.turn_rel( turn_angle )
        else:
            # no lines found, STOP
            back_wheels.stop()

except KeyboardInterrupt:
    print("KeboardInterrupt Captured")
finally:
    detector.stop()
    back_wheels.stop()
    front_wheels.turn_straight()
