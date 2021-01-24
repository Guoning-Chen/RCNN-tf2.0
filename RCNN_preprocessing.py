# offical
from __future__ import division, print_function, absolute_import
import numpy as np
import cv2
import os
import random
# private
import config
import selectivesearch as ss
import tools


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


# Clip Image
def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    # return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]   
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


def generate_ft_or_svm_data(list_path, num_clss, save_path, threshold=0.5,
                            is_svm=False, save=False):
    """"
    按照给定的list文件生成 ft或 svm训练用的数据集。

    :param list_path: path of fine_tune_list.txt.
    :param num_clss: number of class (include background).
    :param save_path: path to save generated example.
    :param threshold: threshold of IoU with ground truth.
    :param is_svm: if true, labels will be scalar instead of one hot vector.
    :param save: if true, save generated data as .npy files to save_path.
    :return: resized RPs (list of float 3D array) and labels (list of scalar or
    one hot).
    """
    fr = open(list_path, 'r')
    train_list = fr.readlines()
    # random.shuffle(train_list)
    for num, line in enumerate(train_list):  # 1 line = 1 image = 1 .npy
        labels = []
        images = []
        tmp = line.strip().split(' ')  # [image path, label, rect GT]
        img = cv2.imread(tmp[0])
        img_lbl, regions = ss.selective_search(img, scale=500, sigma=0.9,
                                               min_size=10)
        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding small regions
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 500:
                continue
            # 按照rect尺寸裁剪原图
            proposal_img, proposal_rect = clip_pic(img, r['rect'])
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
            # resize RPs to the input size of CNN
            resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE,
                                                config.IMAGE_SIZE)
            candidates.add(r['rect'])
            img_float = np.asarray(resized_proposal_img, dtype="float32")
            images.append(img_float)
            # IOU
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IOU(ref_rect_int, proposal_rect)
            # attach labels according to IoU threshold, 0: background
            index = int(tmp[1])
            if is_svm:
                if iou_val < threshold:
                    labels.append(0)  # negative example
                else:
                    labels.append(index)  # positive example
            else:  # fine tune
                label = np.zeros(num_clss + 1)  # one hot
                if iou_val < threshold:
                    label[0] = 1  # negative
                else:
                    label[index] = 1  # positive
                labels.append(label)
        tools.view_bar(
            "processing image of %s" % list_path.split('\\')[-1].strip(),
            num + 1, len(train_list))
        if save:
            np.save((os.path.join(
                save_path, tmp[0].split('/')[-1].split('.')[0].strip())
                     + '_data.npy'), [images, labels])
    print(' ')
    fr.close()


def load_from_npy(data_folder):
    """
    从加载指定文件夹下.npy文件中加载 RPs和对应的标签。

    :param data_folder: .npy文件所在的文件夹。
    :return:
        images: list of 3D arrays, resized RPs
        labels: list of scalar (ft data) or 1D array (SVM data), labels of RPs
    """
    images, labels = [], []
    data_list = os.listdir(data_folder)
    # random.shuffle(data_list)
    for index, npy_name in enumerate(data_list):
        img, label = np.load(os.path.join(data_folder, npy_name),
                             allow_pickle=True)
        images.extend(img)
        labels.extend(label)
        tools.view_bar("load data of %s" % npy_name, index + 1, len(data_list))
    print(' ')
    return images, labels
