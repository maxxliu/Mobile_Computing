import lcm
from lcm_types.mcomp_msgs import event_msg
from time import sleep

lc = lcm.LCM()


msg = event_msg()
msg.timestamp = 0
msg.position = (1, 2, 3)
msg.description = "I saw a UFO"

while(True):
	lc.publish("MCOMP_EVENT", msg.encode())
	sleep(0.5)
