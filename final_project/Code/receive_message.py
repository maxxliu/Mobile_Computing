import lcm
from lcm_types.mcomp_msgs import event_msg

lc = lcm.LCM()

def my_handler(channel, data):
	msg = event_msg.decode(data)
	print "I received a message, it says: '%s'" % str(msg.description)

subscription = lc.subscribe("MCOMP_EVENT", my_handler)

try:
	while True:
		lc.handle()
except KeyboardInterrupt:
	pass
