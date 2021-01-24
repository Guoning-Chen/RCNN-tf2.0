# config path and files

# general settings
IMAGE_SIZE = 224
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
PLOT_FOLDER = './plot'

# pre-train
TRAIN_CLASS = 17
EXAMPLE_LIST_file = './example_list.txt'
TRAIN_LIST = 'train_list'
VAL_LIST = 'val_list'
TEST_LIST = 'test_list'
PRETRAIN_WEIGHT_PATH = './pre_train_model'

# fine tune
FINE_TUNE_CLASS = 3  # 2 classes and background
FINE_TUNE_LIST = './fine_tune_list.txt'
FINE_TUNE_DATASET = './fine_tune_dataset'
FINE_TUNE_WEIGHT = './fine_tune_model'

# svm
SVM = './svm_train'


# chain classify
LIST_PATH = 'C:/Users/chengn/Documents/Project/chain_classify/list/17flowers'

