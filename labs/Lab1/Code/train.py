# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Monday, January 22nd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Thursday, January 25th 2018


from utils import *
import numpy as np
from machine_learning import *

# define constants
features = [ 'xAccl', 'yAccl', 'zAccl', 'time' ]
data_dir = "../Data"            # directory containing the dataset
timesteps_per_sample = 500      # number of IMU readings in one sample
batch_size = 8                  # number of traces used in parallel to train the model
rnn_state_size = 50             # size of the memory of the RNN cells
num_classes = 4                 # classes to chose from (i.e., Standing, Walking, Jumping, Driving)

# compute params
num_features = len(features)    # number of features used by the classifier

# get data
datasets = load_data( data_dir, 'train', timesteps_per_sample, features, verbose=True )

plot_sample( datasets['Jumping'][0], 3, [0,1,2] )


# get model
# get_model( timesteps_per_trace, num_features, rnn_state_size, num_classes, batch_size )
