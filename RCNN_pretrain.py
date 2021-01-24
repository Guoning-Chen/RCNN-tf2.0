from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import codecs
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
import os

import config


def load_pretrain_dataset(datafile, show_examples=False):
    """
    Divide the data set to three parts (train, val, test) by a ratio of 6:2:2
    :param datafile: a text file containing all images' name and label
    :param show_examples: if true, show an image example for each part
    :return: data sets after splitting, [(x_train, y_train), (x_val, y_val),
    (x_test, y_test)] where x_* is 4D array and y_* is 1D array
    """
    fr = codecs.open(datafile, 'r', 'utf-8')
    example_list = fr.readlines()
    num_total = len(example_list)

    # split indices to three parts
    index = np.arange(num_total)
    np.random.shuffle(index)
    index_train = index[:int(0.6 * num_total)]  # 6/10
    index_val = index[int(0.6 * num_total): int(0.8 * num_total)]  # 2/10
    index_test = index[int(0.8 * num_total):]  # 2/10
    indexes = [index_train, index_val, index_test]

    # lists to store data
    x_train, y_train, x_val, y_val, x_test, y_test = [[] for _ in range(6)]
    x_data = [x_train, x_val, x_test]
    y_data = [y_train, y_val, y_test]

    # three text files containing the respective sample names
    ftrain = open(config.TRAIN_LIST, 'w')
    fval = open(config.VAL_LIST, 'w')
    ftest = open(config.TEST_LIST, 'w')
    files = [ftrain, fval, ftest]

    is_first = False  # use for show examples
    names = ["train", "val", "test"]
    for (name, index, x, y, file) in zip(names, indexes, x_data, y_data, files):
        print('{} set: {} examples'.format(name, len(index)))
        if show_examples:
            is_first = True
        for i in index:
            line = example_list[i]
            file.write(line)
            tmp = line.strip().split(' ')  # temp: [image path, label]
            image_path = tmp[0]
            img = cv2.imread(image_path)
            img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            # if is_first:
            #     cv2.imshow(name + " example", img)
            #     cv2.waitKey(1000)
            #     is_first = False
            np_img = np.asarray(img, dtype="float32")
            x.append(np_img)

            label = int(tmp[1])
            y.append(label)  # label：0 ~ 17
        file.close()
    fr.close()

    data_split = []
    for x, y in zip(x_data, y_data):
        data_split.append((np.asarray(x, dtype="float32"),
                           np.asarray(y, dtype="float32")))

    return data_split


class Dataset(object):
    """
    实例化之后的对象被调用时会返回一个以 (x_batch, y_batch) 为迭代对象的迭代器
    x_batch, y_batch 均为第一维度大小为 batch_size 的 np 数组
    """
    def __init__(self, X, y, batch_size):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size = batch_size

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))


def alexnet(num_classes, drop=0.0):
    '''
    Create and return an AlexNet model

    :param num_classes: if is zero, return an AlexNet without the last layer
    which serves as a feature extractor
    :param drop: ratio of neurons to drop
    :return: a tf.keras.Model object
    '''
    # define initialization approach
    initializer = tf.initializers.VarianceScaling(scale=2.0)

    # define layers
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE)
    conv1 = tf.keras.layers.Conv2D(96, 11, 4, kernel_initializer=initializer,
                                   activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPool2D(3, 2, padding='valid')(conv1)
    lrn1 = tf.nn.lrn(pool1)
    conv2 = tf.keras.layers.Conv2D(256, 5, kernel_initializer=initializer,
                                   activation='relu')(lrn1)
    pool2 = tf.keras.layers.MaxPool2D(3, 2, padding='valid')(conv2)
    lrn2 = tf.nn.lrn(pool2)
    conv3 = tf.keras.layers.Conv2D(384, 3, kernel_initializer=initializer,
                                   activation='relu')(lrn2)
    conv4 = tf.keras.layers.Conv2D(384, 3, kernel_initializer=initializer,
                                   activation='relu')(conv3)
    conv5 = tf.keras.layers.Conv2D(384, 3, kernel_initializer=initializer,
                                   activation='relu')(conv4)
    pool3 = tf.keras.layers.MaxPool2D(3, 2, padding='valid')(conv5)
    lrn3 = tf.nn.lrn(pool3)
    flatten = tf.keras.layers.Flatten()(lrn3)
    fc6 = tf.keras.layers.Dense(4096, activation='tanh',
                                kernel_initializer=initializer)(flatten)
    drop1 = tf.keras.layers.Dropout(drop)(fc6)
    fc7 = tf.keras.layers.Dense(4096, activation='tanh',
                                kernel_initializer=initializer)(drop1)

    # decide the last layer
    if num_classes is 0:
        model = tf.keras.Model(inputs=inputs, outputs=fc7)
    else:
        drop2 = tf.keras.layers.Dropout(drop)(fc7)
        fc8 = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    kernel_initializer=initializer,
                                    name='{}classes'.format(num_classes))(drop2)
        model = tf.keras.Model(inputs=inputs, outputs=fc8)

    return model


def train_custom(model, train_set, val_set, test_set=None, lr=0.01, epoch=1,
                 val_every=10, details=False, to_save=False, tb=False):
    """
    Train a model with given data and hyperparameters

    :param train_set: a Dataset object
    :param val_set: a Dataset object
    :param test_set: a Dataset object
    :param val_every: validate every val_every steps
    :param details: if true, print loss values and accuracies of training and
    validating every val_every steps
    :param to_save: path to save weights after training
    :param tb: if true, save training and validation accuracies as tensorboard
    logs to logs/GradientTape/
    :return: None
    """
    device = '/device:GPU:0'

    with tf.device(device):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        # metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

        # tensorboard writer
        if tb:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/GradientTape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            val_log_dir = 'logs/GradientTape/' + current_time + '/val'
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # values for plotting accuracy curves
        train_loss_values = []
        train_acc_values = []
        val_acc_values = []

        steps = 0
        for epoch in range(epoch):
            # reset metrics
            train_loss.reset_states()
            train_accuracy.reset_states()

            # start training
            for x_np, y_np in train_set:
                with tf.GradientTape() as tape:
                    scores = model(x_np, training=True)
                    loss = loss_fn(y_np, scores)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    # back-propagation
                    optimizer.apply_gradients(zip(gradients,
                                                  model.trainable_variables))

                    # update metrics
                    train_loss.update_state(loss)
                    train_accuracy.update_state(y_np, scores)

                    # write training accuracy to tensorboard logs
                    if tb:
                        with train_summary_writer.as_default():
                            tf.summary.scalar("accuracy",
                                              train_accuracy.result(),
                                              step=int(0.25*steps))

                    # validate every print_every steps
                    if steps % val_every == 0:
                        # record training loss and accuracy for plotting
                        train_loss_values.append(train_loss.result())
                        train_acc_values.append(train_accuracy.result())

                        val_loss.reset_states()
                        val_accuracy.reset_states()
                        for val_x, val_y in val_set:
                            prediction = model(val_x, training=False)
                            loss = loss_fn(val_y, prediction)

                            # update validation metrics
                            val_loss.update_state(loss)
                            val_accuracy.update_state(val_y, prediction)

                        # record validation accuracy for plotting
                        val_acc_values.append(val_accuracy.result())

                        # write validation accuracy to tensorboard logs
                        if tb:
                            with val_summary_writer.as_default():
                                tf.summary.scalar('accuracy',
                                                  val_accuracy.result(),
                                                  step=int(steps))

                        # print some information
                        if details:
                            template = 'Iteration {}, Epoch {}, Loss: {:.3f}' \
                                       ', Accuracy: {:.2f}%, Val Loss: {:.3f}' \
                                       ', Val Accuracy: {:.2f}%'
                            print(template.format(
                                steps, epoch + 1, train_loss.result(),
                                train_accuracy.result() * 100,
                                val_loss.result(), val_accuracy.result() * 100))
                    steps += 1

        # test
        if test_set is not None:
            print('Testing model...')
            test_accuracy.reset_states()
            for test_x, test_y in test_set:
                prediction = model(test_x, training=False)
                test_accuracy.update_state(test_y, prediction)
            test_acc = test_accuracy.result()
            print("Test accuracy: {:.2f}%".format(100 * test_acc))

        # save weights
        if to_save:
            weight_file_name = 'pretrain_weight_{:.2f}.h5'.format(100 * test_acc)
            model.save_weights(os.path.join(
                config.PRETRAIN_WEIGHT_PATH, weight_file_name))

        # save and show accuracy curves
        plt.figure()
        x_data = [val_every * x for x in range(len(train_acc_values))]
        plt.plot(x_data, train_acc_values, color="red", label="Train accuracy")
        plt.plot(x_data, val_acc_values, color="green", label="Valid accuracy")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy / %')
        plt.title("Test accuracy: {:.2f}%".format(100*test_acc))
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(config.PLOT_FOLDER,
                                 'train_val_acc_%.2f.jpg' % (100 * test_acc)))
        plt.show()


def train_auto(train_set, val_set, test_set=None, lr=0.01, batch=32, drop=0.0,
               epochs=1, verbose=2, save=False):
    '''
    使用内置训练函数 fit 进行训练
    :param verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param save: if True, save model to config.SAVE_MODEL_PATH
    :return: 训练完成时的验证集准确率
    '''
    x_train_data, y_train_data = train_set
    x_val_data, y_val_data = val_set
    model = alexnet(17, drop=drop)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])

    # 设置 tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 histogram_freq=1)

    history = model.fit(x_train_data, y_train_data, batch_size=batch,
                        epochs=epochs, verbose=verbose,
                        validation_data=(x_val_data, y_val_data),
                        callbacks=[tb_callback])

    if test_set is not None:
        x_test_data, y_test_data = test_set
        print("testing model...")
        _, acc = model.evaluate(x_test_data, y_test_data, batch_size=16,
                                verbose=0)
        print("test accuracy: {:.2f}".format(acc))
    if save:
        model.save_weights('./checkpoints/pretrain_weight')

    return history.history['val_sparse_categorical_accuracy'][-1]


def visual_1d(lr_range, results):
    '''
    条形图
    :param lr_range: 列表, 单位为 10^
    :param results: 列表
    '''
    plt.bar(np.log10(lr_range), results, width=0.1)
    plt.xlabel('10^lr')
    plt.ylabel('val_acc')
    plt.title('learning rate')
    plt.subplots_adjust(left=0.15)
    plt.show()


def visual_2d(lr_range, mu_range, results):
    '''
    散点图
    :param lr_range: 列表, 单位为 10^
    :param mu_range: 列表, 单位为 1
    :param results: 字典，键值对为 (lr, mu): acc
    '''
    x_scatter = [math.log10(x) for x in lr_range]
    y_scatter = mu_range

    # 找出要标记的最优点
    values = results.values()  # acc
    best_acc = max(values)
    max_acc_index = values.index(best_acc)  # results中最大值的位置
    best_point = (lr_range(max_acc_index), mu_range(max_acc_index))  # 最优的(lr, mu)

    plt.figure()
    plt.scatter(x_scatter,  # 以学习率 lr 作为横坐标
                y_scatter,  # 以动量 mu 作为纵坐标
                marker_size=100,
                c=values,  # 以准确率决定点的颜色
                cmap=plt.cm.coolwarm)  # 颜色模式
    plt.annotate('(%.2f,%.2f,%.2f%%)' % (best_point[0], best_point[1], best_acc * 100),  # 文本内容
                 xy=best_point, xytext=(-30, 30), textcoords='offset pixels',  # 确定文本位置
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),  # 文本框样式
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))  # 箭头样式
    plt.colorbar()  # 开启颜色条
    plt.xlabel('lr / 10^')
    plt.ylabel('mu')
    plt.title('choose (lr, mu)')
    plt.show()


def augment_data(x_data):
    """
    :param x_data: 4D array of shape (batch, height, width, channels)
    :return: data after augmentation, list of 4D arrays
    """
    x_flip_up_down = tf.image.flip_up_down(x_data)
    x_flip_left_right = tf.image.flip_left_right(x_data)

    return [x_flip_up_down, x_flip_left_right]


if __name__ == '__main__':
    data = load_pretrain_dataset(config.EXAMPLE_LIST_file, show_examples=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data

    # training set augmentation
    augmented_x_train = augment_data(x_train)
    for x in augmented_x_train:
        x_train = np.concatenate((x_train, x))
    y_train = np.concatenate([y_train for _ in range(len(augmented_x_train) + 1)])
    print('after {}X augmentation，there are {} examples for training'.format(
        len(augmented_x_train) + 1, len(y_train)))

    # split data set to batches
    batch_size = 16
    train_batches = Dataset(x_train, y_train, batch_size=batch_size)
    val_batches = Dataset(x_val, y_val, batch_size=batch_size)
    test_batches = Dataset(x_test, y_test, batch_size=batch_size)

    # start training
    net_pretrain = alexnet(17, drop=0.5)
    train_custom(net_pretrain, train_batches, val_batches, test_batches,
                 lr=0.01, epoch=15, details=True, to_save=True)

    # mu_range = [0.5, 0.9, 0.95, 0.99]
    # lr_range = np.logspace(-3.5, -1.5, num=9)
    # results = []
    # for lr in lr_range:
    #     print('lr = ', lr)
    #     val_np = np.zeros((3, ))
    #     for i in range(3):
    #         val_np[i] = train_auto(x_train, y_train, x_val, y_val,
    #         lr=lr, drop=0.5, epochs=3, verbose=0)
    #     results.append(val_np.mean())
    #
    # visual_1d(lr_range, results)
    # train_auto(train_data, val_data, test_data, lr=pow(10, -2.5),
    # drop=0.5, epochs=5, verbose=2, save=True)
