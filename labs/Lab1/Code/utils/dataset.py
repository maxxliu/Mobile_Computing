# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Monday, January 22nd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Thursday, January 25th 2018

import os
from os.path import isfile, join
import re
import ast
from utils import ProgressBar
import matplotlib.pyplot as plt
import numpy as np
import json
import math

import scipy.fftpack


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


'''
Apply FFT to remove high frequency components of the IMU data

@param sample_data: a list of numpy arrays. Each array is a sample of shape (T, 1, F),
    where T is the time horizon, and F is the number of features.

@param features: a list of strings. Each string identifies a feature (e.g., xAccl, time).
    The feature 'time' must be one of the features.

@return a numpy array of shape (T, 1, F) in which the high frequency components of
    the signal are removed.
'''
def remove_noise( sample_data, features ):
    T, _, F = sample_data.shape
    time_index = features.index('time')
    t_0 = sample_data[0, 0, time_index]
    t_1 = sample_data[1, 0, time_index]
    resolution = t_1 - t_0

    for f in range(F):
        if( features[f] == 'time' ): continue
        y = sample_data[:, 0, f]
        # compute FT of the feature f
        w = scipy.fftpack.rfft(y)
        # compute mean frequency
        mean = np.mean( np.abs(w) )
        # set the threshold to double the mean
        thr = 2 * mean
        # remove high frequency components
        cutoff_idx = np.abs(w) < thr
        w[cutoff_idx] = 0
        # return to time domain by doing inverseFFT
        y = scipy.fftpack.irfft(w)
        sample_data[:, 0, f] = y
    # return filtered sample
    return sample_data


def normalize_dataset( datasets ):
    return datasets


'''
Each txt file has the form (note that it is not a valid JSON file due to the single quotes)

    [                                   # this is a list of all the traces from all the teams
        {                               # this is one trace from one team
            'type': 'Jumping',
            'seq': [
                {
                    'data':{
                        'xGyro': #,
                        'zAccl': #,
                        'yGyro': #,
                        'zGyro': #,
                        'xAccl': #,
                        'xMag': #,
                        'yMag': #,
                        'zMag': #,
                        'yAccl': #
                    },
                    'time': #           # this is the absolute time in seconds
                },
                ...
            ]
        },
        ...
    ]

'''
def load_data( data_dir, data_type, seconds_per_sample, features, use_noise_reduction=True, use_data_normalization=True, activities=None, verbose=False ):
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
    if verbose:
        status = {True : 'ENABLED', False : 'DISABLED'}
        print 'Loading data:'
        print '\tINFO: FFT-based noise reduction %s' % status[use_noise_reduction]
        print '\tINFO: Data normalization %s' % status[use_data_normalization]
        print '\tProgress: ',
    else:
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
    # if we assume that the IMU publish rate is perfect, then there is a fixed
    # relation between seconds_per_sample and readings_per_sample, let's find it
    txt_file_0 = txt_files[0]
    activity_0 = raw_data[ txt_file_0 ]
    trace_0 = activity_0[0]['seq']
    datapoints_count = 0
    start_time = trace_0[0]['time']
    for datapoint in trace_0:
        datapoints_count += 1
        if datapoint['time'] - start_time >= seconds_per_sample: break
    readings_per_sample = datapoints_count
    if verbose:
        print 'INFO: %d seconds/sample corresponds to %d readings/sample @ %.1f Hz' % (
            seconds_per_sample,
            readings_per_sample,
            float(readings_per_sample)/float(seconds_per_sample)
        )
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
            # reduce noise (if enabled)
            if use_noise_reduction:
                # we expect the activity lo leave a unique fingerprint in the IMU readings
                # but we know that IMUs are noisy. Let's remove that noise
                for sample in samples:
                    sample = remove_noise( sample, features )
                    # append sample to the dataset
                    datasets[trace_type].append( sample )
            else:
                # append samples to the dataset
                datasets[trace_type].extend( samples )
            # update progress bar
            pbar.next()
    # normalize data (if enabled)
    if use_data_normalization:
        datasets = normalize_dataset( datasets )
    if verbose:
        # print statistics about the datasets loaded
        print 'INFO: Datasets:'
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
