# official
from __future__ import division, print_function, absolute_import
import os.path
import numpy as np
# private
import config
import RCNN_preprocessing as prep
import RCNN_pretrain as pretrain


def load_ft_data(batch_size=32):
    # load fine tune data from files
    if len(os.listdir(config.FINE_TUNE_DATASET)) == 0:
        print("Generate Fine Tune Data...")
        prep.generate_ft_or_svm_data(config.FINE_TUNE_LIST, 2, save=True,
                                     save_path=config.FINE_TUNE_DATASET)
    print("Load Existed Fine Tune Data...")
    x_data, y_data = prep.load_from_npy(config.FINE_TUNE_DATASET)
    assert len(x_data) == len(y_data), 'ERROR: Numbers of data and labels ' \
                                       'are not equal!'

    # convert to numpy arrays
    x_np, y_np = np.asarray(x_data), np.asarray(y_data)
    # np.random.shuffle(x_np)
    # np.random.shuffle(y_np)

    # convert one-hot label to scalar
    y_np = np.argmax(y_np, axis=1)

    # split them to three parts
    num_total = len(x_np)
    x_train = x_np[:int(0.6 * num_total)]  # 6/10
    y_train = y_np[:int(0.6 * num_total)]
    x_val = x_np[int(0.6 * num_total): int(0.8 * num_total)]  # 2/10
    y_val = y_np[int(0.6 * num_total): int(0.8 * num_total)]
    x_test = x_np[int(0.8 * num_total):]  # 2/10
    y_test = y_np[int(0.8 * num_total):]

    # generate batches
    train_set = pretrain.Dataset(x_train, y_train, batch_size=batch_size)
    val_set = pretrain.Dataset(x_val, y_val, batch_size=batch_size)
    test_set = pretrain.Dataset(x_test, y_test, batch_size=batch_size)

    return train_set, val_set, test_set


if __name__ == '__main__':
    # load fine tune data
    train_batches, val_batches, test_batches = load_ft_data(batch_size=32)

    # construct a new model
    net_fine_tune = pretrain.alexnet(config.FINE_TUNE_CLASS, drop=0.5)

    # load weights of the pre-trained model
    pretrain_weight_list = os.listdir(config.PRETRAIN_WEIGHT_PATH)
    assert pretrain_weight_list is not None, 'ERROR: can not find ' \
                                             'pre-trained weights!'
    pretrain_weight = os.path.join(config.PRETRAIN_WEIGHT_PATH,
                                   pretrain_weight_list[-1])  # the latest one
    net_fine_tune.load_weights(pretrain_weight, by_name=True)

    # start to fine tune
    pretrain.train_custom(net_fine_tune, train_batches, val_batches,
                          test_batches, lr=1e-3, details=True, epoch=3,
                          to_save=config.FINE_TUNE_WEIGHT)
