# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Monday, January 22nd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Sunday, January 28th 2018


from utils import *
import numpy as np
from machine_learning import *

# define constants
features = [ 'xAccl', 'yAccl', 'zAccl', 'time' ]
data_dir = "../Data"            # directory containing the dataset
seconds_per_sample = 2          # number of seconds of IMU readings in one sample
batch_size = 6                  # number of traces used in parallel to train the model
rnn_state_size = 50             # size of the memory of the RNN cells
num_classes = 4                 # classes to chose from (i.e., Standing, Walking, Jumping, Driving)
use_data_shuffling = True       # TODO
use_noise_reduction = True      # whether to use FFT(Fast Fourier Transform) to remove noise
use_data_normalization = True   # whether to normalize the features values to the range [0,1]
learning_rate = 0.001           #
max_epochs = 10                 #
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
    use_data_shuffling,
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
    use_data_shuffling,
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
timesteps, _, _ = train_data['input'][0].shape
X, Y, zero_state, loss, train_op = get_model(
    timesteps,
    num_features,
    rnn_state_size,
    num_classes,
    forward_only=False,
    learning_rate=learning_rate
)




# initialize model
session = tf.Session()
session.run(
    tf.global_variables_initializer()
)


# for memory efficiency, prepare all the possible zero states for any possible batch_size
zero_states = [
    np.zeros( [bsize, rnn_state_size] )
    for bsize in range(batch_size+1)
]

# train the model
cross_validation_num_batches = int( math.floor( len(train_batches['input']) * 0.2) ) # use 20% of training for cross-validation
for epoch in range(1, max_epochs+1, 1):

    epoch_train_losses = []
    epoch_eval_losses = []
    crossval_i = 1

    for cross_train, cross_eval in cross_validation(train_batches, cross_validation_num_batches):
        train_input, train_output = cross_train
        eval_input, eval_output = cross_eval
        n_train_batches = len(train_input)
        n_eval_batches = len(eval_input)

        pbar = ProgressBar( 2*n_train_batches+n_eval_batches )
        print '\nTraining cross-Validation step %d.%d :: ' % ( epoch, crossval_i ),

        # train on all the batches
        for j in range(n_train_batches):
            batch_input = train_input[j]
            batch_output = train_output[j]
            # retrieve initial state for the RNN based on the current batch size
            _, cur_batch_size, _ = batch_input.shape
            rnn_zero_state = zero_states[ cur_batch_size ]
            # train the network on the current batch
            session.run(
                train_op,
                { X : batch_input, Y : batch_output, zero_state : rnn_zero_state }
            )
            # update progress bar
            pbar.next()

        # compute training loss after the current cross-validation iteration
        crossval_losses = []
        for j in range(n_train_batches):
            batch_input = train_input[j]
            batch_output = train_output[j]
            # create initial state for the RNN based on the current batch size
            _, cur_batch_size, _ = batch_input.shape
            rnn_zero_state = zero_states[ cur_batch_size ]
            # feed the batch to the RNN and get the loss
            batch_loss_val = session.run(
                loss,
                { X : batch_input, Y : batch_output, zero_state : rnn_zero_state }
            )
            # store loss
            crossval_losses.append( batch_loss_val )
            # update progress bar
            pbar.next()
        training_loss = np.mean( crossval_losses )
        epoch_train_losses.append( training_loss )

        # compute validation loss after the current cross-validation iteration
        crossval_losses = []
        for j in range(n_eval_batches):
            batch_input = eval_input[j]
            batch_output = eval_output[j]
            # create initial state for the RNN based on the current batch size
            _, cur_batch_size, _ = batch_input.shape
            rnn_zero_state = zero_states[ cur_batch_size ]
            # feed the batch to the RNN and get the loss
            batch_loss_val = session.run(
                loss,
                { X : batch_input, Y : batch_output, zero_state : rnn_zero_state }
            )
            # store loss
            crossval_losses.append( batch_loss_val )
            # update progress bar
            pbar.next()
        evaluation_loss = np.mean( crossval_losses )
        epoch_eval_losses.append( evaluation_loss )

        # print some stats and increment counters
        print 'Cross-Validation step %d.%d :: Training loss: %.2f \t Validation loss: %.2f' % (
            epoch, crossval_i, training_loss, evaluation_loss
        )
        crossval_i += 1

    # print some stats
    epoch_train_loss = np.mean( epoch_train_losses )
    epoch_eval_loss = np.mean( epoch_eval_losses )
    print 'Epoch %d :: Training loss: %.2f \t Validation loss: %.2f' % (
        epoch, epoch_train_loss, epoch_eval_loss
    )
