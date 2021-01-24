import math
import sys
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import config
import os
import codecs


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def show_rect(img_path, regions, caption=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    if caption is not None:
        plt.title(caption)
    plt.show()


def load_data_set(flag, num_classes, show_example=False):
    '''
    加载训练集/验证集/测试集，并以 numpy 数组形式返回

    :param flag: 0 ~ 2分别代表加载训练集、验证集、测试集
    :param num_classes: 要加载的类别数量
    :param show_example: 若为真，每个类别展示一张
    :return: x, y都是 4D 数组
    '''
    assert flag >= 0 & flag <= 2, 'ERROR(load_data_set): flag范围为 0 ~ 2'
    part_names = ['train', 'val', 'test']
    part_name = part_names[flag]
    class_names = os.listdir(config.LIST_PATH)
    class_names = class_names[:num_classes]
    img_nums = []  # 记录每个类分别加载了多少图像
    x, y = [], []
    for class_name in class_names:
        class_path = os.path.join(config.LIST_PATH, class_name)
        target_list_path = class_path + '/' + class_name + '_' + part_name +  '_list.txt'
        assert os.path.isfile(target_list_path), '找不到 class %s 的 %s_list.txt文件' % (class_name, part_name)
        list_file = codecs.open(target_list_path, 'r', 'utf-8')
        lines = list_file.readlines()
        img_nums.append(len(lines))
        print('Loading %d examples of class %s to %s' % (len(lines), class_name, part_name))
        if show_example:
            to_show = True
        else:
            to_show = False
        for line in lines:
            img_path, label = line.strip().split(' ')
            img = cv2.imread(img_path)
            if to_show:
                cv2.imshow('example of class %s to %s' % (class_name, part_name), img)
                cv2.waitKey(1000)
                to_show = False
            np_img = np.asarray(img, dtype="float32")
            np_img = np.transpose(np_img, (2, 0, 1))  # 由 W*H*C转置为 C*W*H，以适应pytorch的输入
            x.append(np_img)
            assert class_name == label, 'ERROR(load_data_set): list中图像的标签和文件夹名称不一致'
            y.append(label)
        list_file.close()
    assert len(x) == len(y), '数据和标签的个数不一致'
    print('Finish loading %d images(' % len(x), end='')
    for i, num in enumerate(img_nums):
        print('class %d: %d, ' % (i, num), end='')
    print(') to %s!' % part_name)
    x = np.asarray(x, dtype="float32")
    np.random.shuffle(x)
    y = np.asarray(y, dtype="float32")
    np.random.shuffle(y)
    return x, y