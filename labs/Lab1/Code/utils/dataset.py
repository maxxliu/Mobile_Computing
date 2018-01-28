# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Monday, January 22nd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Friday, January 26th 2018

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
import random

# if we assume that the IMU publish rate is perfect, then there is a fixed
# relation between seconds_per_sample and readings_per_sample
IMU_FREQUENCY = 128             # average frequency computed on the dataset


def _compute_features( sample_data, features ):
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

def _extract_features_from_sample( sample_data, features ):
    T = len(sample_data)    # T = timesteps
    F = len(features)       # F = number of features
    # extract features from sample
    sample_features = np.zeros( dtype=np.float32, shape=(T, 1, F) )
    _compute_features( sample_data, features )
    # copy features' values
    for t in range(T):
        datapoint = sample_data[t]['data']
        for f in range(F):
            feature_key = features[f]
            sample_features[t, 0, f] = datapoint[ feature_key ]
    # return features
    return sample_features


def _extract_samples( trace_data, seconds_per_sample, features ):
    readings_per_sample = IMU_FREQUENCY * seconds_per_sample
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
        sample_features = _extract_features_from_sample( sample_data, features )
        # extend the existing dataset
        output.append( sample_features )
    # return samples
    return output

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
def load_data( data_dir, data_type, seconds_per_sample, features, verbose=False ):
    # generate classes map
    idx_to_class = { -1 : 'Unknown', 0 : 'Standing', 1 : 'Walking', 2 : 'Jumping', 3 : 'Driving' }
    class_to_idx = { 'Unknown' : -1, 'Standing' : 0, 'Walking' : 1, 'Jumping' : 2, 'Driving' : 3 }
    # create pattern for file
    regex = "activity-dataset-%s([0-9]+).txt" % data_type
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
    print '\nLoading data [%s]: ' % data_type,
    pbar = ProgressBar( num_txt_files )
    # get content of the JSON files
    raw_data = {}
    for txt_file in txt_files:
        # read file content
        json_content = None
        with open(join(data_dir, data_type, txt_file), 'r') as fopen:
            json_content = fopen.read();
            json_content = json_content.replace( '\'', '"' )
        # convert JSON to python dict
        activity_data = json.loads( json_content )
        raw_data[ txt_file ] = activity_data
        # update progress bar
        pbar.next()
    # do post-processing on the raw data
    num_traces = sum([ len(a) for a in raw_data.values() ])
    print 'Pre-Processing data: ',
    pbar = ProgressBar( num_traces )
    # create dataset
    dataset = {
        'input' : [],
        'output' : []
    }
    # iterate over the txt files' content
    for activity in raw_data.values():
        # iterate over the traces
        for trace in activity:
            trace_type = trace['type']
            trace_seq = trace['seq']
            # fragment the trace data into a sequence of smaller samples
            samples = _extract_samples( trace_seq, seconds_per_sample, features )
            for sample in samples:
                # append sample to the dataset
                dataset['input'].append( sample )
                dataset['output'].append( class_to_idx[trace_type] )
            # update progress bar
            pbar.next()
    # make sure input and output are the same size
    assert( len(dataset['input']) == len(dataset['output']) )
    # shuffle data
    indices = range( len(dataset['input']) )
    random.shuffle( indices )
    random.shuffle( indices )
    dataset['input'] = [ dataset['input'][i] for i in indices ]
    dataset['output'] = [ dataset['output'][i] for i in indices ]
    # print some statistics (if needed)
    if verbose:
        # print statistics about the dataset loaded
        print '[INFO :: Data Loader] : Dataset size: %d samples' % len(dataset['input'])
    # return dataset
    return idx_to_class, class_to_idx, dataset


'''
We expect the activity lo leave a unique fingerprint in the IMU readings
but we know that IMUs are noisy.
This function applies FFT to remove the high frequency components of the IMU readings.
This function works in place and returns None.

@param dataset: a dataset dictionary of the form
    {
        'input' : [ ... ],
        'output' : [ ... ]
    }
    where 'input' is a list of numpy arrays. Each array is a sample of shape (T, 1, F),
    where T is the time horizon, and F is the number of features. The value of 'output'
    is not used.

@param features: a list of strings. Each string identifies a feature (e.g., xAccl, time).
    The feature 'time' must be one of the features.
'''
def remove_noise( dataset, features, verbose=False ):
    sample_0 = dataset['input'][0]
    _, _, F = sample_0.shape
    print 'Removing noise: ',
    pbar = ProgressBar( len(dataset['input'])*(F-1)  )
    for f in range(F):
        if( features[f] == 'time' ): continue # no need to filter time
        for sample in dataset['input']:
            y = sample[:, 0, f]
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
            sample[:, 0, f] = y
            # update progress bar
            pbar.next()
    # return
    return None


'''
Normalize readings to the range [0,1].
This function works in place.

@param dataset: a dataset dictionary of the form
    {
        'input' : [ ... ],
        'output' : [ ... ]
    }
    where 'input' is a list of numpy arrays. Each array is a sample of shape (T, 1, F),
    where T is the time horizon, and F is the number of features. The value of 'output'
    is not used.

@param feature_max (optional): a list of floats. feature_max[i] contains the max value
    of the i-th in the training data. If not passed, feature_max will be computed on
    the current data.

@param feature_min (optional): a list of floats. feature_min[i] contains the min value
    of the i-th in the training data. If not passed, feature_min will be computed on
    the current data.

@return: a tuple (feature_max, feature_min).
'''
def normalize_dataset( dataset, feature_max=None, feature_min=None, verbose=False ):
    sample_0 = dataset['input'][0]
    _, _, F = sample_0.shape
    # compute feature_max and feature_min if not passed
    if( feature_max == None or feature_min == None ):
        if verbose:
            print '[INFO :: Data Normalization] : Either feature_max or feature_min was not passed. They will be recomputed.'
        # create placeholders for max and min of each feature
        print 'Normalizing dataset: ',
        pbar = ProgressBar( 2 * len(dataset['input']) * F  )
        feature_max = [ sample_0[0,0,f] for f in range(F) ]
        feature_min = [ sample_0[0,0,f] for f in range(F) ]
        # compute max value for each feature
        for f in range(F):
            for sample in dataset['input']:
                # compute max and min values for the current sample
                sample_max = np.amax( sample[:,0,f] )
                sample_min = np.amin( sample[:,0,f] )
                # update feature's max/min
                if sample_max > feature_max[f]: feature_max[f] = sample_max
                if sample_min < feature_min[f]: feature_min[f] = sample_min
                # update progress bar
                pbar.next()
    else:
        if verbose:
            print '[INFO :: Data Normalization] : feature_max and feature_min are passed. They will NOT be recomputed.'
            print 'Normalizing dataset: ',
            pbar = ProgressBar( len(dataset['input']) * F  )
    # normalize dataset
    for f in range(F):
        for sample in dataset['input']:
            sample[:,0,f] = ( sample[:,0,f] - feature_min[f] ) / ( feature_max[f] - feature_min[f] )
            # update progress bar
            pbar.next()
    # return features boundaries
    return ( feature_max, feature_min )


'''
Partitions the given dataset in data batches of size (at most) batch_size.

@param dataset: a dataset dictionary of the form
    {
        'input' : [ ... ],
        'output' : [ ... ]
    }
    where 'input' is a list of numpy arrays each with shape (T, 1, F), where T is the time horizon,
    and F is the number of features; 'output' is a list of integers in which the i-th element
    indicates the class id of the i-th input in 'input'.

@param batch_size: the number of samples in each batch. The last batch can be smaller
    than the others depending on the total number of samples and the value of batch_size.

@return: a dataset dictionary of the form
    {
        'input' : [ ... ],
        'output' : [ ... ]
    }
    where 'input' is a list of numpy arrays each with shape (T, batch_size, F),
    where T is the time horizon, and F is the number of features; 'output' is a list of numpy arrays
    each with shape (batch_size,) containing the ground-truth classes for each batch in 'input'.
'''
def batchify( dataset, batch_size ):
    # compute number of batches
    total_samples = len( dataset['input'] )
    batch_size = min( batch_size, total_samples )
    num_batches = int( math.ceil( float(total_samples) / float(batch_size) ) )
    # create buckets
    buckets = [ (l-batch_size, l) for l in range(batch_size, total_samples+1, batch_size) ]
    if( total_samples % batch_size != 0 ):
        buckets.append( (buckets[-1][1], total_samples) )
    # iterate over the buckets and extract batches
    batched_dataset = {
        'input' : [],
        'output' : []
    }
    for bucket in buckets:
        i, f = bucket
        # create batch
        batch_input = np.concatenate( dataset['input'][i:f], axis=1 )
        batch_output = np.asarray( dataset['output'][i:f] )
        # append batch
        batched_dataset['input'].append( batch_input )
        batched_dataset['output'].append( batch_output )
    # return data batches
    return batched_dataset


'''
Iterates over the data batches and returns validation and training set using the cross-validation
technique.

@param batches: a dataset dictionary of the form
    {
        'input' : [ ... ],
        'output' : [ ... ]
    }
    where 'input' is a list of numpy arrays each with shape (T, batch_size, F),
    where T is the time horizon, and F is the number of features; 'output' is a list of numpy arrays
    each with shape (batch_size,) containing the ground-truth classes for each batch in 'input'.

@param validation_batches: an integer indicating how many batches will constitute the validation set
    at each iteration.

@return: an iterator over tuples. Each tuple contains two elements, training and validation batches,
    respectively. Training and validation batches are stored in tuples of two elements, respectively,
    list of numpy arrays of shape (T, batch_size, F) representing the batch input, and list of numpy
    arrays of shape (batch_size,) representing the batch output.

    Example:

        at the i-th iteration:

            (
                (
                    [ ... ],        # list of numpy arrays with training batches inputs
                    [ ... ]         # list of numpy arrays with training batches output
                ),
                (
                    [ ... ],        # list of numpy arrays with validation batches inputs
                    [ ... ]         # list of numpy arrays with validation batches output
                )
            )
'''
def cross_validation( batches, validation_batches ):
    total_batches = len( batches['input'] )
    validation_batches = min( validation_batches, total_batches )
    # create buckets
    buckets = [ (l-validation_batches, l) for l in range(validation_batches, total_batches+1, validation_batches) ]
    if( total_batches % validation_batches != 0 ):
        buckets.append( (buckets[-1][1], total_batches) )
    # iterate over the buckets and provide training and validation sets
    for bucket in buckets:
        i, f = bucket
        # extract validation batches
        validation_batches = (
            batches['input'][i:f],
            batches['output'][i:f]
        )
        # extract training batches
        training_batches = (
            batches['input'][0:i] + batches['input'][f:],
            batches['output'][0:i] + batches['output'][f:]
        )
        # provide data for current iteration
        yield ( training_batches, validation_batches )



def plot_sample( sample_data, x_axis, y_axes, data_is_normalized ):
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
        if data_is_normalized:
            # adjust Y-limits (if data is normalized)
            plt.ylim( 0, 1 )
        # plot
        plt.plot( x, y, 'r' )
    plt.show()
