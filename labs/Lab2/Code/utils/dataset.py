# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Thursday, February 8th 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Thursday, February 8th 2018

import os
from os.path import isfile, join
import re
from utils import ProgressBar
import numpy as np
import json

def load_data( data_dir ):
    # create pattern for file
    regex = "rss-([0-9]+).txt"
    file_pattern = re.compile( regex );
    # get txt files matching the pattern
    txt_files = [
        f for f in os.listdir(data_dir)
        if  isfile(join(data_dir, f))
            and
            re.match( file_pattern, f )
    ]
    txt_files = sorted( txt_files )
    num_txt_files = len(txt_files)
    # get raw data from disk
    print '\nLoading data: ',
    pbar = ProgressBar( 3*num_txt_files )
    # get content of the JSON files
    raw_data = {}
    for txt_file in txt_files:
        # read file content
        json_content = None
        with open(join(data_dir, txt_file), 'r') as fopen:
            json_content = fopen.read();
        # convert JSON to python dict
        activity_data = json.loads( json_content )
        raw_data[ txt_file ] = activity_data
        # update progress bar
        pbar.next()
    # collect all the MACs present
    macs = set()
    for txt_file in txt_files:
        for datapoint in raw_data[txt_file]:
            macs.add( datapoint['mac'] )
    # create maps: MAC_to_ID and ID_to_MAC
    ID_to_MAC = list(macs)
    MAC_to_ID = { ID_to_MAC[i] : i for i in range(len(macs)) }
    # sort raw data wrt time
    for txt_file in txt_files:
        x = sorted(raw_data[txt_file], key=lambda k: k['time'])
        # let's be crazy enough not to trust python (make sure the datapoints are in order wrt time ASC)
        assert [(x[k+1]['time']-x[k]['time'])>0 for k in range(len(x)-1)].count(True) == len(x)-1
        raw_data[txt_file] = x
        # update progress bar
        pbar.next()
    # convert data into numpy arrays
    data = {}
    features_order = [ 'time', 'mac', 'loc_x', 'loc_y', 'rss' ]
    for txt_file in txt_files:
        trace_id = re.search( regex, txt_file ).group(1)
        num_datapoints = len(raw_data[txt_file])
        trace_arr = np.empty( [num_datapoints, len(features_order)], dtype=np.float64 )
        for i in range(num_datapoints):
            datapoint = raw_data[txt_file][i]
            # replace MAC with ID
            datapoint['mac'] = MAC_to_ID[ datapoint['mac'] ]
            # convert the trace into a numpy array
            datapoint_arr = np.asarray( [ datapoint[f] for f in features_order ] )
            trace_arr[i,:] = datapoint_arr
        # make the time relative
        trace_arr[:,0] -= trace_arr[0,0]
        # store data
        data[trace_id] = trace_arr
        # update progress bar
        pbar.next()
    # return data
    return ID_to_MAC, MAC_to_ID, data


def filter_readings_given_mac( trace_data, mac_id ):
    useful_readings = [ trace_data[i,1] == mac_id for i in range(trace_data.shape[0]) ]
    for r in trace_data[ useful_readings ]:
        yield r
