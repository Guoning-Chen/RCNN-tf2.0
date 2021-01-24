# official
import os
import shutil
# private
import config


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # delete folders recursively
    os.mkdir(path)  # create


if __name__ == '__main__':
    # save data set for fine tuning
    mkdir(config.FINE_TUNE_DATASET)
    # save pre-train weights
    mkdir(config.PRETRAIN_WEIGHT_PATH.strip().rsplit('/', 1)[0])
    # save fine-tune weights
    mkdir(config.FINE_TUNE_WEIGHT.strip().rsplit('/', 1)[0])
    # save svm models and data sets
    mkdir(os.path.join(config.SVM, '1'))
    mkdir(os.path.join(config.SVM, '2'))