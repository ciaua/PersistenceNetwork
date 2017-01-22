import lasagne
import theano.tensor as T
from pnet import utils


if __name__ == '__main__':

    use_real_data = True

    # Data options
    out_param_fp = 'data/MagnaTagATune/sample_model.npz'

    # Downloading data to this dir
    exp_data_dir = 'data/MagnaTagATune/exp_data/'

    # Training options
    lr = 0.01
    loss_function = lasagne.objectives.binary_crossentropy
    n_epochs = 10
    batch_size = 10  # we use 10 for real data
    feat_type = "logmelspec10000.16000_512_512_128.0.standard"
    measure_type = 'mean_auc'

    n_top = 50

    # Network options
    network_type = 'pc_fcn'
    pl_seg_size = 32
    network_options = {
        'early_conv_dict': {
            'conv_filter_list': [(64, 8, 1)],
            'pool_filter_list': [4],
            'pool_stride_list': [None]
        },
        'middle_conv_dict': {
            'conv_filter_list': [(3200, 1, 1)],
            'pool_filter_list': [pl_seg_size],
            'pool_stride_list': [None],
        },
        'pl_dict': {
            'num_lambda': 5,
            'num_points': 10,
            'value_range': (0, 5),
            'seg_size': pl_seg_size,
            'seg_step': pl_seg_size
        },
        'late_conv_dict': {
            'conv_filter_list': [(512, 1), (512, 1)],
            'pool_filter_list': [None, None],
            'pool_stride_list': [None, None],
        },
        'dense_filter_size': 1,
        'final_pool_function': T.mean,
        'input_size': 128,
        'output_size': n_top,
        'p_dropout': 0.5
    }

    # Loading data
    print("Loading data...")
    X_tr, y_tr = utils.load_data('tr', exp_data_dir=exp_data_dir,
                                 use_real_data=use_real_data)
    X_va, y_va = utils.load_data('va', exp_data_dir=exp_data_dir,
                                 use_real_data=use_real_data)

    network, input_var, lr_var, train_func, val_func, pr_func = \
        utils.make_network(
            network_type, loss_function, lr, network_options
        )

    # Training
    utils.train(
        X_tr, y_tr, X_va, y_va,
        network,
        train_func, val_func,
        n_epochs, batch_size, lr_var,
        measure_type=measure_type,
        param_fp=out_param_fp
    )
