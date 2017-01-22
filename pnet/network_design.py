#!/usr/bin/env python

import lasagne
from lasagne import layers
import theano.tensor as T
from pnet import layers as cl


# Parts
def conv_layers(network, conv_dict, total_stride,
                init_input_size=1, p_dropout=0,
                W_list=None, b_list=None, return_params=False,
                base_name=''):
    '''
    conv_dict:
        a dictionary containing the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

        conv_filter_list: list
            each element is in the form (# of filters, filter length)

        pool_filter_list: list
            each element is an integer

        pool_stride_list: list
            each element is int or None

    '''

    conv_filter_list = conv_dict['conv_filter_list']
    pool_filter_list = conv_dict['pool_filter_list']
    pool_stride_list = conv_dict['pool_stride_list']
    assert(len(conv_filter_list) ==
           len(pool_filter_list) ==
           len(pool_stride_list))
    n_layers = len(conv_filter_list)

    out_W_list = []
    out_b_list = []

    # shared variables
    if type(W_list) is list:
        if len(W_list) != n_layers:
            assert(False)
    elif W_list is None:
        W_list = [lasagne.init.GlorotUniform() for kk in range(n_layers)]
    else:
        assert(False)

    if type(b_list) is list:
        if len(b_list) != n_layers:
            assert(False)
    elif b_list is None:
        b_list = [lasagne.init.Constant(0.) for kk in range(n_layers)]
    else:
        assert(False)

    for ii, [conv_filter, pool_filter, pool_stride, W, b] in enumerate(
            zip(conv_filter_list, pool_filter_list, pool_stride_list,
                W_list, b_list)):
        if len(conv_filter) == 2:
            n_filters, filter_len = conv_filter
            conv_stride = 1
            pad = 'strictsamex'
        elif len(conv_filter) == 3:
            n_filters, filter_len, conv_stride = conv_filter
            if conv_stride is None:
                conv_stride = 1

            if conv_stride == 1:
                pad = 'strictsamex'
            else:
                pad = 'valid'
        total_stride *= conv_stride

        if ii == 0:
            feat_dim = init_input_size
        else:
            feat_dim = 1

        network = cl.Conv2DXLayer(
            lasagne.layers.dropout(network, p=p_dropout),
            num_filters=n_filters, filter_size=(filter_len, feat_dim),
            stride=(conv_stride, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad=pad,
            W=W, b=b,
            name='{}.conv{}'.format(base_name, ii)
        )
        out_W_list.append(network.W)
        out_b_list.append(network.b)

        if (pool_filter is not None) or (pool_filter > 1):
            if pool_stride is None:
                stride = None
                total_stride *= pool_filter
            else:
                stride = (pool_stride, 1)
                total_stride *= pool_stride

            # network = lasagne.layers.MaxPool2DLayer(
            network = cl.MaxPool2DXLayer(
                network,
                pool_size=(pool_filter, 1),
                stride=stride,
                ignore_border=False,
                # pad='strictsamex',
                name='{}.maxpool{}'.format(base_name, ii)
            )

    if return_params:
        return network, total_stride, out_W_list, out_b_list
    else:
        return network, total_stride


# Multi-label (4D input)
def fcn(input_var,
        early_conv_dict,
        late_conv_dict,
        dense_filter_size,
        final_pool_function=T.max,
        input_size=128, output_size=188,
        p_dropout=0.5):
    '''
    early_conv_dict: dict
        it contains the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

        pool_filter_list: list
            each element is an integer

        pool_stride_list: list
            each element is int or None

    late_conv_dict: dict
        it contains the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    dense_filter_size: int
        the filter size of the final dense-like conv layer

    '''
    # early conv layers
    input_network = lasagne.layers.InputLayer(
        shape=(None, 1, None, input_size), input_var=input_var)

    total_stride = 1
    network, total_stride = conv_layers(input_network, early_conv_dict,
                                        total_stride,
                                        init_input_size=input_size,
                                        p_dropout=0,
                                        base_name='early')

    # late conv layers (dense layers)
    network, total_stride = conv_layers(network, late_conv_dict,
                                        total_stride,
                                        init_input_size=1,
                                        p_dropout=p_dropout,
                                        base_name='late')

    # frame output layer. every frame has a value
    network = cl.Conv2DXLayer(
        lasagne.layers.dropout(network, p=p_dropout),
        num_filters=output_size, filter_size=(dense_filter_size, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform()
    )

    # pool
    network = layers.GlobalPoolLayer(network,
                                     pool_function=final_pool_function)
    network = layers.ReshapeLayer(network, ([0], -1))

    return network


def p_fcn(input_var,
          early_conv_dict,
          pl_dict,
          late_conv_dict,
          dense_filter_size,
          final_pool_function=T.max,
          input_size=128, output_size=188,
          p_dropout=0.5):
    '''
    This uses PL 2D layer instead 2D layer, comparing to fcn_pnn_multisource

    early_conv_dict_list: list
        each element in the list is a dictionary containing
        the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

        pool_filter_list: list
            each element is an integer

        pool_stride_list: list
            each element is int or None

    pl_dict: dict
        it contains the following keys:
        'num_lambda', 'num_points', 'value_range', 'seg_size', 'seg_stride'

    late_conv_dict: dict
        it contains the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    dense_filter_size: int
        the filter size of the final dense-like conv layer


    '''
    # early conv layers
    input_network = lasagne.layers.InputLayer(
        shape=(None, 1, None, input_size), input_var=input_var)

    total_stride = 1
    network, total_stride = conv_layers(
        input_network, early_conv_dict,
        total_stride,
        init_input_size=input_size,
        p_dropout=0,
        base_name='early')

    # Persistence landscape
    num_lambda = pl_dict['num_lambda']
    num_points = pl_dict['num_points']
    value_range = pl_dict['value_range']
    seg_size = pl_dict['seg_size']
    seg_step = pl_dict['seg_step']

    patch_size = (seg_size, 1)
    patch_step = (seg_step, 1)

    network = cl.PersistenceFlatten2DLayer(
        network,
        num_lambda, num_points, value_range,
        patch_size, patch_step)

    # late conv layers (dense layers)
    network, total_stride = conv_layers(
        network, late_conv_dict,
        total_stride,
        init_input_size=1,
        p_dropout=p_dropout,
        base_name='late')

    # frame output layer. every frame has a value
    network = cl.Conv2DXLayer(
        lasagne.layers.dropout(network, p=p_dropout),
        num_filters=output_size, filter_size=(dense_filter_size, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform()
    )

    # pool
    network = layers.GlobalPoolLayer(network,
                                     pool_function=final_pool_function)
    network = layers.ReshapeLayer(network, ([0], -1))

    return network


def pc_fcn(
        input_var,
        early_conv_dict,
        middle_conv_dict, pl_dict,
        late_conv_dict,
        dense_filter_size,
        final_pool_function=T.max,
        input_size=128, output_size=188,
        p_dropout=0.5
        ):
    '''
    early_conv_dict_list: list
        each element in the list is a dictionary containing
        the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    pl_dict: dict
        it contains the following keys:
        'num_lambda', 'num_points', 'value_range', 'seg_size', 'seg_stride'

    late_conv_dict: dict
        it contains the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    dense_filter_size: int
        the filter size of the final dense-like conv layer

    pool_filter_list: list
        each element is an integer

    pool_stride_list: list
        each element is int or None

    '''
    # early conv layers
    input_network = lasagne.layers.InputLayer(
        shape=(None, 1, None, input_size), input_var=input_var)

    total_stride = 1
    network, total_stride = conv_layers(
        input_network, early_conv_dict,
        total_stride,
        init_input_size=input_size,
        p_dropout=0,
        base_name='early')

    # middle conv layers (dense layers)
    network_c, _ = conv_layers(
        network, middle_conv_dict,
        total_stride,
        init_input_size=1,
        p_dropout=0,
        base_name='middle_conv')

    # Persistence landscape
    network_p = network
    try:
        num_lambda = pl_dict['num_lambda']
    except:
        num_lambda = pl_dict['n_f_db']

    try:
        num_points = pl_dict['num_points']
    except:
        num_points = pl_dict['n_points']

    value_range = pl_dict['value_range']
    seg_size = pl_dict['seg_size']
    seg_step = pl_dict['seg_step']

    patch_size = (seg_size, 1)
    patch_step = (seg_step, 1)

    network_p = cl.PersistenceFlatten2DLayer(
        network_p,
        num_lambda, num_points, value_range,
        patch_size, patch_step)

    # Convolution+Persistence
    network = layers.ConcatLayer(
        [network_c, network_p],
        axis=1,
        cropping=[None, None, 'lower', None],
        name='Convolution+Persistence')

    # late conv layers (dense layers)
    network, total_stride = conv_layers(
        network, late_conv_dict,
        total_stride,
        init_input_size=1,
        p_dropout=p_dropout,
        base_name='late')

    # frame output layer. every frame has a value
    network = cl.Conv2DXLayer(
        lasagne.layers.dropout(network, p=p_dropout),
        num_filters=output_size, filter_size=(dense_filter_size, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform()
    )

    # pool
    network = layers.GlobalPoolLayer(network,
                                     pool_function=final_pool_function)
    network = layers.ReshapeLayer(network, ([0], -1))

    return network
