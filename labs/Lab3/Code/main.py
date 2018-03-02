#Lab3

from mypicar.front_wheels import Front_Wheels
from mypicar.back_wheels import Back_Wheels
from detector_wrapper import DetectorWrapper
import time
import numpy as np

detector = DetectorWrapper()

front_wheels = Front_Wheels()
back_wheels = Back_Wheels()

max_turn = 28
max_speed = 20
turn_gain = .25
turn_offset = -2
x_offset = 0

use_memory = False
memory = []
memory_size = 3

back_wheels.stop()
front_wheels.turn_rel( turn_offset )

i = 0
try:
    while True:
        success, ret = detector.detect()
        detector.plot(ret)

        if success:
            i += 1

            back_wheels.speed = max_speed
            back_wheels.forward()

            frame, mid_x, left_fit, right_fit, ploty, left_fitx, right_fitx = ret


            # mid_x = float(sum(memory)) / float(len(memory))

            img_width = frame.shape[0]

            ideal_mid_x = 300 #float(img_width) / 2.0 + x_offset

            error = mid_x - ideal_mid_x

            turn_angle = turn_offset + error * turn_gain

            if i < 8: continue

            # memory = memory[-memory_size:]
            memory.append( turn_angle )

            if use_memory:
                turn_angle = 0.0
                mem_size = min(len(memory),memory_size)
                for i in range(1,mem_size,1):
                    turn_angle += ( float(mem_size-i+1)/float(mem_size) ) * memory[-i]

            print 'mid-point @ %d pxs / %d pxs, turning @ %.2f' % (mid_x, img_width, turn_angle)

            front_wheels.turn_rel( turn_angle )
            # time.sleep(0.5)
            # front_wheels.turn_rel( turn_offset )
        else:
            back_wheels.stop()


except KeyboardInterrupt:
    print("KeboardInterrupt Captured")
finally:
    detector.stop()
    back_wheels.stop()
    front_wheels.turn_straight()
