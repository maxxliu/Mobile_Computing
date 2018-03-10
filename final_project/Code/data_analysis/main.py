# coding: utf-8

from lcm_types.mcomp_msgs import experiment_log_msg
import pickle
import json


data = pickle.load(open("../../Data/1/03_2018_10_h13_m40/data.dat","r"))
experiment_meta = json.load(open("../../Data/1/03_2018_10_h13_m40/meta.json","r"))

for msg in data:
    msg_timestamp, msg_data_encoded = msg
    msg_data = experiment_log_msg.decode(msg_data_encoded)

    print msg_data.location
