#!/usr/bin/env python

import os
import sys
import csv
import numpy as np
import theano
import theano.tensor as T
import lasagne

from pnet import network_design, measure


theano.config.exception_verbosity = 'high'

floatX = theano.config.floatX
epsilon = np.float32(1e-6)
one = np.float32(1)
pf = np.float32(0.5)


# IO
ver = sys.version_info
if ver >= (3, 0):
    import pickle as pk
    opts_write = {'encoding': 'utf-8', 'newline': ''}
    opts_read = {'encoding': 'utf-8'}
else:
    import cPickle as pk
    opts_write = {}
    opts_read = {}


# IO
def read_lines(file_path):
    with open(file_path, 'r') as opdrf:
        data = [term.strip() for term in opdrf.readlines()]
        return data


def read_csv(file_path):
    with open(file_path, 'r', **opts_read) as opdrf:
        csv_reader = csv.reader(opdrf)
        data = [term for term in csv_reader]
        return data


def pickle(file_path, obj, protocol=2):
    """
    For python 3 compatibility, use protocol 2
    """
    if not file_path.endswith('.pkl'):
        file_path += '.pkl'
    with open(file_path, 'wb') as opdwf:
        pk.dump(obj, opdwf, protocol=protocol)


def unpickle(file_path):
    with open(file_path, 'rb') as opdrf:
        data = pk.load(opdrf)
        return data


# Load data
def load_data(phase,
              exp_data_dir='data/MagnaTagATune/exp_data/',
              feat_type='logmelspec10000.16000_512_512_128.0.standard',
              use_real_data=False):
    '''
    Modified from the code from
    https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
    '''

    if use_real_data:
        import sys
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename,
                     feat_type='logmelspec10000.16000_512_512_128.0.standard',
                     exp_data_dir='data/MagnaTagATune/exp_data/',
                     source='http://mac.citi.sinica.edu.tw/~liu/data/'):
            print("Downloading {}/{}".format(feat_type, filename))
            if not os.path.exists(exp_data_dir):
                os.makedirs(exp_data_dir)
            urlretrieve(
                os.path.join(source, feat_type, filename),
                os.path.join(exp_data_dir, filename)
            )

        def load_magnatagatune_feats(
                filename,
                exp_data_dir='data/MagnaTagATune/exp_data/'):
            exp_data_fp = exp_data_dir+filename
            if not os.path.exists(exp_data_fp):
                download(filename, exp_data_dir=exp_data_dir)

            data = np.load(exp_data_fp)

            # (instances, channels, frames, features)
            return data

        def load_magnatagatune_labels(
                filename,
                exp_data_dir='data/MagnaTagATune/exp_data/'):
            if not os.path.exists(exp_data_dir+filename):
                download(filename, exp_data_dir=exp_data_dir)
            data = np.load(exp_data_dir+filename)
            return data

        X = load_magnatagatune_feats('feat.{}.npy'.format(phase),
                                     exp_data_dir)
        y = load_magnatagatune_labels('target.{}.npy'.format(phase),
                                      exp_data_dir)

    else:
        def load(filename,
                 exp_data_dir='data/MagnaTagATune/sample_exp_data'):
            fp = os.path.join(exp_data_dir, filename)
            data = np.load(fp)

            # (instances, channels, frames, features)
            return data

        X = load('feat.{}.npy'.format(phase))
        y = load('target.{}.npy'.format(phase))

    return X, y


# Make networks
def make_network(
        network_type, loss_function, lr, net_options, do_clip=True):
    target_var = T.matrix('targets')
    lr_var = theano.shared(np.array(lr, dtype=floatX))

    print("Building model and compiling functions...")
    # input_var = T.addbroadcast(T.tensor4('inputs'))
    input_var = T.tensor4('input')
    network = getattr(network_design, network_type)(input_var, **net_options)

    # Compute loss
    prediction = lasagne.layers.get_output(network)
    if do_clip:
        prediction = T.clip(prediction, epsilon, one-epsilon)
    loss = loss_function(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adagrad(
        loss, params, learning_rate=lr_var)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    if do_clip:
        test_prediction = T.clip(test_prediction, epsilon, one-epsilon)
    test_loss = loss_function(test_prediction, target_var)
    test_loss = test_loss.mean()

    train_func = theano.function([input_var, target_var],
                                 loss, updates=updates)
    val_func = theano.function([input_var, target_var],
                               [test_prediction, test_loss])

    pr_func = theano.function([input_var], test_prediction)

    return network, input_var, lr_var, train_func, val_func, pr_func


def make_network_test(
        network_type, net_options, do_clip=True):
    print("Building model and compiling functions...")
    input_var = T.tensor4('input')
    network = getattr(network_design, network_type)(input_var, **net_options)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    if do_clip:
        test_prediction = T.clip(test_prediction, epsilon, one-epsilon)

    pr_func = theano.function([input_var], test_prediction)
    return network, input_var, pr_func


# Iterate inputs
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if type(targets) == np.ndarray:
        n = len(targets)
        k = targets.shape[-1]
        assert len(inputs) == n

    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
    for start_idx in range(0, n - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt].reshape((-1, k))


def iterate_minibatches_feat(inputs, batchsize, shuffle=False):
    n = len(inputs)

    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
    for start_idx in range(0, n - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


# Functions used in train for recording and printing
def check_best_measure(best_va_measure, va_measure):
    if va_measure > best_va_measure:
        best_va_measure = va_measure

        best_va_updated = True
    else:
        best_va_updated = False
    return best_va_measure, best_va_updated


def print_in_train(epoch, n_epochs,
                   mean_tr_loss, mean_va_loss,
                   best_va_epoch, best_va_measure):
    print("Epoch {} of {}.".format(epoch, n_epochs))
    print("  training loss:        {:.6f}".format(mean_tr_loss))
    print("  validation loss:      {:.6f}".format(mean_va_loss))
    print("  best va (epoch, measure):({}, {:.6f})".format(
        best_va_epoch, best_va_measure
    ))
    print(" ")


# Multiple input sources
def train(
        X_tr, y_tr, X_va, y_va,
        network,
        train_func, va_func,
        n_epochs, batch_size, lr_var, measure_type='mean_auc', param_fp=None):

    print("Starting training...")

    best_va_epoch = 0
    # best_va_loss = np.inf
    best_va_measure = 0
    for epoch in range(1, n_epochs+1):
        train_loss = 0
        train_batches = 0

        # Training
        for inputs, targets in iterate_minibatches(X_tr, y_tr,
                                                   batch_size, shuffle=True):
            train_loss_one = train_func(inputs, targets)

            train_loss += train_loss_one
            train_batches += 1
        mean_tr_loss = train_loss/train_batches

        # Validation
        pre_list, mean_va_loss, va_measure = validate(X_va, y_va, va_func,
                                                      measure_type)

        # Check best loss
        best_va_measure, best_va_updated = check_best_measure(
            best_va_measure, va_measure)
        if best_va_updated:
            best_va_epoch = epoch
            if param_fp is not None:
                save_model(param_fp, network)

        # Print the results for this epoch:
        print_in_train(epoch, n_epochs,
                       mean_tr_loss, mean_va_loss,
                       best_va_epoch, best_va_measure)


def validate(X, y, va_func, measure_type):
    va_loss = 0
    va_batches = 0
    pre_list = []
    for inputs, targets in iterate_minibatches(X, y, 1, shuffle=False):
        pre, loss = va_func(inputs, targets)
        va_loss += loss
        va_batches += 1
        pre_list.append(pre)

    mean_va_loss = va_loss / va_batches

    # Measure
    pre = np.vstack(pre_list)

    measure_func = getattr(measure, measure_type)
    va_measure = measure_func(y, pre)

    return pre_list, mean_va_loss, va_measure


def predict(X, pr_func):
    pre_list = []
    for inputs in iterate_minibatches_feat(X, 1, shuffle=False):
        pre = pr_func(inputs)
        pre_list.append(pre)

    return pre_list


def predict_all(X, pr_func):
    pre_list = pr_func(X)

    return pre_list


# Save/load
def save_model(fp, network):
    np.savez(fp, *lasagne.layers.get_all_param_values(network))


def load_model(fp, network):
    with np.load(fp) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)


# Get measure
def get_measure(arg):
    threshold, prediction, target, step_size, lower_b, measure_func = arg
    pred_binary = ((prediction-threshold) > 0).astype(int)

    measures = measure_func(target, pred_binary)
    return measures


# Process tag list
def get_test_tag_indices(tag_tr_fp, tag_te_fp, tag_conv_fp):
    tag_te_list = read_lines(tag_te_fp)
    tag_conv_dict = dict(read_csv(tag_conv_fp))

    tag_tr_list = read_lines(tag_tr_fp)

    tag_idx_list = [tag_tr_list.index(tag_conv_dict[tag])
                    for tag in tag_te_list]
    return tag_idx_list


if __name__ == '__main__':
    x = T.tensor3()
    func = theano.function([x], [2*x])
