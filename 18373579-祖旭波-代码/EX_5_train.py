# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as op
import scipy.io as scio
import os
import gzip


def load_data(data_folder):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz']
    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return x_train, y_train


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_reg(theta_0, train_images_0, train_labels_0, l0_0):
    m, _ = train_images_0.shape
    hx = sigmoid(np.dot(train_images_0, theta_0.T))
    ln_h = np.log(hx)
    part1 = -np.dot(ln_h.T, train_labels_0.T) / m
    ln_1h = np.log(1 - hx)
    part2 = -np.dot(ln_1h.T, 1 - train_labels_0.T) / m
    reg = l0_0 * np.dot(theta_0, theta_0.T) / (2 * m)
    return (part1 + part2 + reg).flatten()


def grad_reg(theta_0, train_images_0, train_labels_0, l0_0):
    m, _ = train_images_0.shape
    theta_tempt = theta_0.copy()
    hx = sigmoid(np.dot(train_images_0, theta_tempt.T))
    part1 = np.dot(train_images_0.T, hx - train_labels_0.T)
    part2 = l0_0 * theta_tempt
    return ((part1 + part2) / m).flatten()


def fmincg(theta_0, train_images_0, train_labels_0, l0_0, num_labels_0):
    for i in range(num_labels_0):
        y_tempt = train_labels_0.copy()
        pos = np.where(train_labels_0 == i)
        neg = np.where(train_labels_0 != i)
        y_tempt[pos] = 1
        y_tempt[neg] = 0
        result_0 = op.minimize(cost_reg, theta_0[i], args=(train_images_0, y_tempt, l0_0),
                               method="TNC", jac=grad_reg)
        theta_0[i] = result_0.x
    return theta_0


def train():
    # ================== 导入数据 =============================================================
    train_images, train_labels = load_data('MNIST_data/')
    train_images = np.array([im.reshape(784) for im in train_images])
    num_labels = 10
    # ============================ 训练模型 ===========================================
    l0 = 0.001
    num = 100
    m, n = train_images.shape
    theta = np.zeros((num_labels, n))
    for i in range(num):
        print(i)
        left = i * m / num
        right = (i + 1) * m / num
        theta = fmincg(theta, train_images[int(left):int(right)], train_labels[int(left):int(right)], l0, num_labels)
    scio.savemat('dataNew.mat', {'theta': theta})
    return 0
