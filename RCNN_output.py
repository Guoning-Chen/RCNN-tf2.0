from __future__ import division, print_function, absolute_import
import numpy as np
import os.path
import os
import cv2
import joblib
from sklearn import svm

import config
import selectivesearch
import RCNN_preprocessing as prep
from RCNN_pretrain import alexnet
import tools


def image_proposal(img_path):
    """
    Produce region proposals for a given image

    :param img_path: absolute path of an image
    :return:
        images: list of 3D arrays
        vertices: list of tuples (left, top, right, bottom)
    """
    img = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(
                       img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding small regions
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        # resize to 227 * 227 for input
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
        # Delete Empty array
        if len(proposal_img) == 0:
            continue
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = prep.resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices


def generate_single_svm_train_data(svm_train_list):
    """
    prepare training data for binary SVM
    :param svm_train_list: training list of one class
    :return: a list of RPs (3D array) and a list of labels (scalar)
    """
    save_path = svm_train_list.rsplit('.', 1)[0].strip()
    if len(os.listdir(save_path)) == 0:
        print("Generate %s's svm dataset..." % svm_train_list.split('\\')[-1])
        prep.generate_ft_or_svm_data(svm_train_list, 2, save_path,
                                     threshold=0.3, is_svm=True, save=True)
    print("Load svm dataset...")
    images, labels = prep.load_from_npy(save_path)

    return images, labels


def train_svms(svm_folder, feature_extractor):
    """
    根据svm_folder下的训练列表（1.txt和2.txt）训练级联的svm分类器，并保存同一文件夹。
    :param svm_folder: path of the folder which includes all files for svm
    :param feature_extractor: AlexNet without the last layer
    :return: list of trained cascade svm classifiers
    """
    files = os.listdir(svm_folder)
    svm_classifiers = []
    for file_name in files:
        if file_name.split('.')[-1] == 'txt':
            train_list_path = os.path.join(svm_folder, file_name)
            X, Y = generate_single_svm_train_data(train_list_path)
            train_features = []
            for ind, i in enumerate(X):
                # extract features
                i = np.expand_dims(i, 0)  # 4D array of shape (1, 224, 224, 3)
                featrues = feature_extractor.predict([i])
                train_features.append(featrues[0])
                tools.view_bar("extract features of %s" % file_name, ind + 1, len(X))
            print(' ')
            print("feature dimension: ", np.shape(train_features))
            # train SVM
            binary_svm = svm.LinearSVC()
            print("fit svm")
            binary_svm.fit(train_features, Y)
            svm_classifiers.append(binary_svm)
            joblib.dump(binary_svm, os.path.join(
                svm_folder, str(file_name.split('.')[0]) + '_svm.pkl'))
    return svm_classifiers


if __name__ == '__main__':
    # an AlexNet without the last layer which serves as feature extractor
    cnn = alexnet(num_classes=0, drop=0.5)

    # load fine tuned weights
    weight_list = os.listdir(config.FINE_TUNE_WEIGHT)
    assert weight_list is not None, 'ERROR: can not find fine-tune weights!'
    ft_weight_path = os.path.join(config.FINE_TUNE_WEIGHT, weight_list[-1])
    cnn.load_weights(ft_weight_path, by_name=True)  # Attention "by_name=True"

    # load the cascade svm classifiers
    svm_folder = config.SVM
    svms = []
    print("Load existed svm classifiers...", end="")
    for file in os.listdir(svm_folder):
        if file.split('_')[-1] == 'svm.pkl':
            svms.append(joblib.load(os.path.join(svm_folder, file)))
    if len(svms) == 0:
        print("train svm classifiers...")
        svms = train_svms(svm_folder=svm_folder, feature_extractor=cnn)
    print("done!")

    # draw RPs for the test image
    test_img = './17flowers/jpg/16/image_1336.jpg'
    rps, rects = image_proposal(test_img)
    tools.show_rect(test_img, rects, caption='Proposed regions')

    # extract features for every rp and send them to svm classifiers
    features = cnn.predict(np.asarray(rps))
    print("predict image: ", np.shape(features))
    rect_results = []
    label_results = []
    index = 0  # index of current feature
    for f in features:
        for svm in svms:
            pred = svm.predict([f.tolist()])
            if pred[0] != 0:  # not background
                rect_results.append(rects[index])
                label_results.append(pred[0])
        index += 1
    print("result:")
    for rect, label in zip(rect_results, label_results):
        print(rect, label)
    tools.show_rect(test_img, rect_results, caption='inference')
