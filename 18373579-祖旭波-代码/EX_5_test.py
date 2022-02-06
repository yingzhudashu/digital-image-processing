# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
import os
import gzip


def load_data(data_folder):
    files = ['t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))
    with gzip.open(paths[0], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return x_test, y_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pred_lr(theta_0, train_images_0):
    train_images_0 = train_images_0.reshape((1, -1))
    result_0 = sigmoid(train_images_0 @ theta_0.T)
    predict = np.argmax(result_0.flatten())
    return predict


def pred_accuracy(theta_0, train_images_0, train_labels_0):
    right_num = 0
    m, _ = train_images_0.shape
    for i in range(m):
        pred = pred_lr(theta_0, train_images_0[i])
        if pred == train_labels_0[i]:
            right_num += 1
    return right_num / m


def test(num):
    # ================== 导入数据 =============================================================
    test_images, test_labels = load_data('MNIST_data/')
    test_images = np.array([im.reshape(784) for im in test_images])
    # ================== 识别预测 =============================================================
    num = num-1
    data = scio.loadmat('dataNew.mat')
    theta = data['theta']
    result = pred_lr(theta, test_images[int(num)])
    accuracy = pred_accuracy(theta, test_images, test_labels)
    test_image = np.array(test_images[int(num)].reshape((28, 28)))
    test_images = np.array([im.reshape((28, 28)) for im in test_images])
    scio.savemat('test.mat', {'test_images': test_images, 'test_image': test_image,
                              'result': result, 'accuracy': accuracy, 'label': test_labels[int(num)]})
    return 0


def test_n(num):
    # ================== 导入数据 =============================================================
    _, test_labels = load_data('MNIST_data/')
    num = num-1
    data = scio.loadmat('dataNew.mat')
    theta = data['theta']
    data_n = scio.loadmat('dataNew_n.mat')
    test_images = data_n['test_image_n']
    test_images = np.array([im.reshape(784) for im in test_images])
    result = pred_lr(theta, test_images[int(num)])
    accuracy = pred_accuracy(theta, test_images, test_labels)
    test_image = np.array(test_images[int(num)].reshape((28, 28)))
    scio.savemat('test_n.mat', {'test_image': test_image, 'result': result, 'accuracy': accuracy})
    return 0
