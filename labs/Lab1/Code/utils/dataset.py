# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Monday, January 22nd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Wednesday, January 24th 2018

import os
from os.path import isfile, join
import re
import ast
from utils import ProgressBar
import matplotlib.pyplot as plt
import numpy as np
import json
import math


def compute_features( sample_data, features ):
    raw_features = sample_data[0]['data'].keys()
    adj_features = set(features) - set(raw_features)
    # return if there is no adjoint feature
    if len(adj_features) == 0: return
    # compute adjoint features
    if( 'time' in adj_features ):
        # subtract min_time to make the timing relative
        min_time = min( [s['time'] for s in sample_data] )
        for s in sample_data:
            s['data']['time'] = s['time'] - min_time
    #
    #TODO

def extract_features_from_sample( sample_data, features ):
    T = len(sample_data)    # T = timesteps
    F = len(features)       # F = number of features
    # extract features from sample
    sample_features = np.zeros( dtype=np.float32, shape=(T, 1, F) )
    compute_features( sample_data, features )
    # copy features' values
    for t in range(T):
        datapoint = sample_data[t]['data']
        for f in range(F):
            feature_key = features[f]
            sample_features[t, 0, f] = datapoint[ feature_key ]
    # return features
    return sample_features


def extract_samples( trace_data, readings_per_sample, features ):
    output = []
    readings_this_trace = len( trace_data )
    # compute the number of complete samples in the current trace
    num_complete_samples = int( math.floor(readings_this_trace/readings_per_sample) )
    for i in range(num_complete_samples):
        # get datapoints of the current sample
        start = i * readings_per_sample
        end = (i+1) * readings_per_sample
        sample_data = trace_data[ start : end ]
        # extract features of the current points
        sample_features = extract_features_from_sample( sample_data, features )
        # extend the existing dataset
        output.append( sample_features )
    # return samples
    return output


def load_data( data_dir, data_type, readings_per_sample, features, activities=None, verbose=False ):
    if( activities == None ):
        activities = "Standing|Walking|Jumping|Driving"
    else:
        activities = '|'.join(activities)
    # create pattern for file
    regex = "activity-dataset-(%s).txt" % ( activities )
    file_pattern = re.compile( regex );
    # get txt files matching the pattern
    txt_files = [
        f for f in os.listdir(join(data_dir, data_type))
        if  isfile(join(data_dir, data_type, f))
            and
            re.match( file_pattern, f )
    ]
    num_txt_files = len(txt_files)
    # get raw data from disk
    print 'Loading data: ',
    pbar = ProgressBar( num_txt_files )
    # get content of the JSON files
    raw_data = {}
    activities = set()
    for txt_file in txt_files:
        # read file content
        json_content = None
        with open(join(data_dir, data_type, txt_file), 'r') as fopen:
            json_content = fopen.read();
            json_content = json_content.replace( '\'', '"' )
        # convert JSON to python dict
        activity_data = json.loads( json_content )
        raw_data[ txt_file ] = activity_data
        # append activity to set of activities
        activities = activities | set([ d['type'] for d in activity_data ])
        # update progress bar
        pbar.next()
    # do post-processing on the raw data
    num_traces = sum([ len(a) for a in raw_data.values() ])
    print 'Processing data: ',
    pbar = ProgressBar( num_traces )
    # create different datasets for different activities
    datasets = { act : [] for act in activities }
    # iterate over the txt files' content
    for activity in raw_data.values():
        # iterate over the traces
        for trace in activity:
            trace_type = trace['type']
            trace_seq = trace['seq']
            # fragment the trace data into a sequence of smaller samples
            samples = extract_samples( trace_seq, readings_per_sample, features )
            # append samples to the dataset
            datasets[trace_type].extend( samples )
            # update progress bar
            pbar.next()
    if verbose:
        # print statistics about the datasets loaded
        print 'Datasets:'
        for activity_type, dataset in datasets.items():
            print '\t%s: %d samples' % ( activity_type, len(dataset) )
    # return datasets
    return datasets


def plot_sample( sample_data, x_axis, y_axes ):
    num_features = len(y_axes)
    # create figure window
    plt.figure(1)
    # create plots
    for i in range( num_features ):
        # create subplot
        fig_num = i+1
        plot_pos = 310 + fig_num
        plt.subplot(plot_pos)
        # get X series
        x = sample_data[:, 0, x_axis].flatten()
        # get Y series
        y_axis = y_axes[i]
        y = sample_data[:, 0, y_axis].flatten()
        # plot
        plt.plot( x, y, 'r' )
    plt.show()
