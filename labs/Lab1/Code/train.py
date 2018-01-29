# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Monday, January 22nd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Monday, January 29th 2018


from utils import *
import numpy as np
from machine_learning import *
from time import gmtime, strftime

# define constants
features = [ 'xAccl', 'yAccl', 'zAccl', 'time' ]
data_dir = "../Data"            # directory containing the dataset
logs_dir = "../tensorboard_logs"
models_dir = "../models"
seconds_per_sample = 1          # number of seconds of IMU readings in one sample
decimation_factor = 8           # each sample retains one datapoint every decimation_factor datapoints
trace_trim_secs = 2             # do not consider the first and last trace_trim_secs seconds of the trace
batch_size = 6                  # number of traces used in parallel to train the model
rnn_state_size = 50             # size of the memory of the RNN cells
num_classes = 4                 # classes to chose from (i.e., Standing, Walking, Jumping, Driving)
use_data_shuffling = True       # whether to shuffle the samples
use_noise_reduction = True      # whether to use FFT(Fast Fourier Transform) to remove noise
use_data_normalization = True   # whether to normalize the features values to the range [0,1]
learning_rate = 0.001           # learning rate to use for training the network
max_epochs = 50                 # maximum number of epochs to train the model for
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

# create unique label for this run
run_descriptor = {
    'nfeat' : num_features,
    'df' : decimation_factor,
    'sps' : seconds_per_sample,
    'trim' : trace_trim_secs,
    'B' : batch_size,
    'H' : rnn_state_size,
    'shuff' : int(use_data_shuffling),
    'fft' : int(use_noise_reduction),
    'norm' : int(use_data_normalization),
    'lr' : learning_rate
}
keys_order = ['nfeat','df','sps','trim','B','H','shuff','fft','norm','lr']
model_label = '%s-%s' % (
    strftime("%Y-%m-%d-%H:%M", gmtime()),
    '-'.join( [ '%s_%s' % (k,run_descriptor[k]) for k in keys_order ] )
)


# get training data
idx_to_class, class_to_idx, train_data = load_data(
    data_dir,
    'train',
    seconds_per_sample,
    trace_trim_secs,
    decimation_factor,
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
    trace_trim_secs,
    decimation_factor,
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
    print '[INFO :: Model Training] : Data shuffling %s' % status[use_data_shuffling]


# create the model
timesteps, _, _ = train_data['input'][0].shape
X, Y, Y_star, zero_state, loss, train_op = get_model(
    timesteps,
    num_features,
    rnn_state_size,
    num_classes,
    forward_only=False,
    learning_rate=learning_rate
)

# Add ops to save and restore all the variables.
saver = tf.train.Saver( tf.all_variables(), max_to_keep=9999 )
checkpoint_path = '%s/%s/model_ckpt_ep' % (models_dir, model_label)

# initialize model
session = tf.Session()
session.run(
    tf.global_variables_initializer()
)


# enable tensorboard
writer = tf.summary.FileWriter(
    '%s/%s/' % (logs_dir, model_label),
    graph=tf.get_default_graph()
)
epoch_performance_phold = tf.placeholder(tf.float32, (), 'performance_per_epoch')
iter_performance_phold = tf.placeholder(tf.float32, (), 'performance_per_iteration')
epoch_performance_summ = tf.summary.scalar("performance_per_epoch", epoch_performance_phold)
iter_performance_summ = tf.summary.scalar("performance_per_iteration", iter_performance_phold)


# for memory efficiency, prepare all the possible zero states for any possible batch_size
zero_states = [
    np.zeros( [bsize, rnn_state_size] )
    for bsize in range(batch_size+1)
]


if verbose:
    # print stats about the size of the neural network
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print '[INFO :: Model Training] : Total trainable parameters: %d' % total_parameters


# train the model
epoch_train_accuracy = 0.0
epoch_eval_accuracy = 0.0
cross_validation_num_batches = int( math.floor( len(train_batches['input']) * 0.2) ) # use 20% of training for cross-validation
global_iteration = 0
for epoch in range(1, max_epochs+1, 1):

    epoch_train_losses = []
    epoch_eval_losses = []
    epoch_train_correct = 0
    epoch_eval_correct = 0
    epoch_total_train_samples = 0
    epoch_total_eval_samples = 0
    crossval_i = 1

    for cross_train, cross_eval in cross_validation(train_batches, cross_validation_num_batches):
        train_input, train_output = cross_train
        eval_input, eval_output = cross_eval
        n_train_batches = len(train_input)
        n_eval_batches = len(eval_input)
        n_train_samples = 0
        n_eval_samples = 0

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
                { X : batch_input, Y_star : batch_output, zero_state : rnn_zero_state }
            )
            # update progress bar
            pbar.next()

        # compute training loss after the current cross-validation iteration
        crossval_losses = []
        training_correct = 0
        for j in range(n_train_batches):
            batch_input = train_input[j]
            batch_output = train_output[j]
            # create initial state for the RNN based on the current batch size
            _, cur_batch_size, _ = batch_input.shape
            rnn_zero_state = zero_states[ cur_batch_size ]
            # feed the batch to the RNN and get the loss
            Y_pdist, batch_loss_val = session.run(
                [Y, loss],
                { X : batch_input, Y_star : batch_output, zero_state : rnn_zero_state }
            )
            # store loss
            crossval_losses.append( batch_loss_val )
            # compute accuracy
            Y_val = np.argmax( Y_pdist, axis=1 )
            correct = np.sum( Y_val == batch_output )
            training_correct += correct
            n_train_samples += cur_batch_size
            # update progress bar
            pbar.next()
        training_loss = np.mean( crossval_losses )
        epoch_train_losses.append( training_loss )
        epoch_train_correct += training_correct

        # compute validation loss after the current cross-validation iteration
        crossval_losses = []
        evaluation_correct = 0
        for j in range(n_eval_batches):
            batch_input = eval_input[j]
            batch_output = eval_output[j]
            # create initial state for the RNN based on the current batch size
            _, cur_batch_size, _ = batch_input.shape
            rnn_zero_state = zero_states[ cur_batch_size ]
            # feed the batch to the RNN and get the loss
            Y_pdist, batch_loss_val = session.run(
                [Y, loss],
                { X : batch_input, Y_star : batch_output, zero_state : rnn_zero_state }
            )
            # store loss
            crossval_losses.append( batch_loss_val )
            # compute accuracy
            Y_val = np.argmax( Y_pdist, axis=1 )
            correct = np.sum( Y_val == batch_output )
            evaluation_correct += correct
            n_eval_samples += cur_batch_size
            # update progress bar
            pbar.next()
        evaluation_loss = np.mean( crossval_losses )
        epoch_eval_losses.append( evaluation_loss )
        epoch_eval_correct += evaluation_correct

        epoch_total_train_samples += n_train_samples
        epoch_total_eval_samples += n_eval_samples

        training_accuracy = 100.*float(training_correct)/float(n_train_samples)
        evaluation_accuracy = 100.*float(evaluation_correct)/float(n_eval_samples)

        # print some stats and increment counters
        print 'Cross-Validation step %d.%d :: Training loss: %.2f (%.1f%%) \t Validation loss: %.2f (%.1f%%)' % (
            epoch, crossval_i,
            training_loss, training_accuracy,
            evaluation_loss, evaluation_accuracy
        )

        # publish data on tensorboard
        summ = session.run(
            iter_performance_summ,
            { iter_performance_phold : evaluation_accuracy }
        )
        writer.add_summary( summ, global_iteration )
        writer.flush()

        global_iteration += 1
        crossval_i += 1

    # print some stats
    epoch_train_loss = np.mean( epoch_train_losses )
    epoch_eval_loss = np.mean( epoch_eval_losses )
    epoch_train_accuracy = 100.*float(epoch_train_correct)/float(epoch_total_train_samples)
    epoch_eval_accuracy = 100.*float(epoch_eval_correct)/float(epoch_total_eval_samples)
    print 'Epoch %d :: Training loss: %.2f (%.1f%%) \t Validation loss: %.2f (%.1f%%)' % (
        epoch,
        epoch_train_loss, epoch_train_accuracy,
        epoch_eval_loss, epoch_eval_accuracy
    )

    # publish data on tensorboard
    summ = session.run(
        epoch_performance_summ,
        { epoch_performance_phold : epoch_eval_accuracy }
    )
    writer.add_summary( summ, epoch )
    writer.flush()

    # store weights
    saver.save( session, checkpoint_path, global_step=epoch )
