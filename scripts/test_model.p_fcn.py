import os
from pnet import utils, measure
import theano.tensor as T
import numpy as np


if __name__ == '__main__':
    # Options
    model_type = 'P-FCN.P_5'
    network_type = 'pc_fcn'

    test_measure_type = 'auc_y_classwise'
    # test_measure_type = 'ap_y_classwise'

    # Dirs and fps
    model_dir = os.path.join('data/MagnaTagATune/models', model_type)

    # Downloading data to this dir
    exp_data_dir = 'data/MagnaTagATune/exp_data/'

    print(model_type)

    # Network options
    network_type = 'p_fcn'
    pl_seg_size = 32
    network_options = {
        'early_conv_dict': {
            'conv_filter_list': [(64, 8, 1)],
            'pool_filter_list': [4],
            'pool_stride_list': [None]
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
        'output_size': 50,
        'p_dropout': 0.5
    }

    # Load data
    X_te, y_te = utils.load_data('te', exp_data_dir=exp_data_dir,
                                 use_real_data=True)

    # Make network
    network, input_var, pr_func = utils.make_network_test(
        network_type, network_options
    )

    fn_list = sorted(os.listdir(model_dir))
    score_list = list()
    for fn in fn_list:
        print('Processing model {}'.format(fn))
        param_fp = os.path.join(model_dir, fn)

        # Load params
        utils.load_model(param_fp, network)

        # Predict
        pred_list_raw = utils.predict(
            X_te,
            pr_func
        )
        pred_all_raw = np.vstack(pred_list_raw)

        pred_all = pred_all_raw
        anno_all = y_te

        # out_dict = dict()
        measure_func = getattr(measure, test_measure_type)
        score = measure_func(anno_all, pred_all)
        print('Model {}: {}'.format(fn, score.mean()))

        score_list.append(score.mean())
    print('Average: {}'.format(np.mean(score_list)))
