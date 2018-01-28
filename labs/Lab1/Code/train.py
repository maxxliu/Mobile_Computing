# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Monday, January 22nd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Sunday, January 28th 2018


from utils import *
import numpy as np
# from machine_learning import *

# define constants
features = [ 'xAccl', 'yAccl', 'zAccl', 'time' ]
data_dir = "../Data"            # directory containing the dataset
seconds_per_sample = 2          # number of seconds of IMU readings in one sample
batch_size = 6                  # number of traces used in parallel to train the model
rnn_state_size = 50             # size of the memory of the RNN cells
num_classes = 4                 # classes to chose from (i.e., Standing, Walking, Jumping, Driving)
use_noise_reduction = False     # whether to use FFT(Fast Fourier Transform) to remove noise
use_data_normalization = True   # whether to normalize the features values to the range [0,1]
verbose = True                  # enables the verbose mode


# header
print 'Mobile Computing :: Lab1 :: Team 8'
print 'Andrea F. Daniele, Max X. Liu, Noah A. Hirsch\n'


# compute params
num_features = len(features)    # number of features used by the classifier


# report which options are enabled
if verbose:
    status = {True : 'ENABLED', False : 'DISABLED'}
    print '[INFO :: Model Training] : FFT-based noise reduction %s' % status[use_noise_reduction]
    print '[INFO :: Model Training] : Data normalization %s' % status[use_data_normalization]


# get training data
idx_to_class, class_to_idx, train_data = load_data(
    data_dir,
    'train',
    seconds_per_sample,
    features,
    verbose
)
if use_noise_reduction:     # remove noise by applying FFT (if needed)
    remove_noise( train_data, features, verbose )
if use_data_normalization:  # apply data normalization as regularization technique
    feature_max, feature_min = normalize_dataset( train_data, verbose=verbose )
train_batches = batchify( train_data, batch_size )


# get test data
_, _, test_data = load_data(
    data_dir,
    'test',
    seconds_per_sample,
    features,
    verbose
)
if use_noise_reduction:     # remove noise by applying FFT (if needed)
    remove_noise( test_data, features, verbose )
if use_data_normalization:  # apply data normalization as regularization technique
    normalize_dataset( test_data, feature_max, feature_min, verbose )
test_batches = batchify( test_data, batch_size )


# print statistics about data batches
if verbose:
    print
    print '[INFO :: Model Training] : Training data: %d batches' % len( train_batches['input'] )
    print '[INFO :: Model Training] : Test data: %d batches' % len( test_batches['input'] )


# create the model




# train the model
i = 0
for cross_train, cross_eval in cross_validation(train_batches, 3):
    train_input, train_output = cross_train
    eval_input, eval_output = cross_eval

    print 'Iteration %d' % i
    print '%d training batches' % len( train_input )
    print '%d validation batches' % len( eval_input )
    print

    i += 1



# plot_sample( train_data['input'][0], 3, [0,1,2], use_data_normalization )
# print idx_to_class[ train_data['output'][0] ]
