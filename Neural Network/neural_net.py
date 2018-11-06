#!~/anaconda3/bin/ python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Implement a 3 layers neural network and train and test on MNIST dataset
"""

import sys
import random
import logging
import numpy as np
import matplotlib.pyplot as plt

D = 28  # image height and width
H = 200  # hidden layer size
O = 10  # number of output classes


class NeuralNetwork(object):

    def __init__(self, D, H, O, max_iter):
        self.D = D
        self.H = H
        self.O = O
        self.w1 = np.random.normal(0, 0.01, (self.H, self.D * self.D + 1))
        self.w2 = np.random.normal(0, 0.01, (self.O, self.H + 1))
        self.max_iter = max_iter
        self.correct_num = 0
        self.acc = []

    def trainNN(self, images, labels):
        """
        Train Neural Network (1 Hidden Layer)

        Argument:
            images: [D, D, N_train] matrix, first two dimensions are image dimensions, and the last dimension
                    is the number of training samples
            labels: [N_train, 1] vector, each row as a label

        Return:
            self.W1: [H, (D*D+1)] matrix, the weights (and bias) between the input layer and hidden layer
            self.w2: [O, (H+1)] matrix, the weights (and bias) between the hidden layer and output layer
        """

        for _iter in range(1, self.max_iter + 1):
            # randomly pick samples
            idx = random.randint(0, images.shape[0] - 1)
            # back propagation
            grad_j, grad_k, correct = self.grad(images, labels, idx)
            self.w1 -= 0.01 * grad_j
            self.w2 -= 0.01 * grad_k
            self.correct_num += correct
            self.acc.append(self.correct_num / images.shape[0])
            logger.info("Iteration {}, picked sample {}".format(_iter, idx))

        return self.w1, self.w2

    def testNN(self, images, labels):
        """
        Test Neural Network (1 Hidden Layer)

        Argument:
            images: [D, D, N_test] matrix,first two dimensions are image dimensions, and the last dimension
                    is the number of testing samples
            labels: [N_test, 1] vector, each row as a label
            self.w1: [H, (D*D+1)] matrix, the weights (and bias) between the input layer and hidden layer
            self.w2: [O, (H+1)] matrix, the weights (and bias) between the hidden layer and output layer

        Return:
            accuracy: real-valued scalar, accuracy of the neural network for the dataset
        """
        count = 0

        for idx, sample in enumerate(images):

            sample = np.asarray(images[idx]).reshape(-1, 1)

            # add bias in input layer
            sample = np.concatenate((sample, np.array([[1]])), axis=0).reshape(1, -1)
            uj = np.dot(self.w1, sample.T).reshape(self.H, 1)
            uj = np.concatenate((uj, np.array([[1]])), axis=0)

            # add bias in hidden layer
            yj = self.tanh(uj).reshape(-1, 1)
            uk = np.dot(self.w2, yj).reshape(self.O, 1)
            zk = self.sigmoid(uk).tolist()

            # calculate accuracy
            res = zk.index(max(zk))
            if res == labels[idx][0]:
                count += 1
            print("true label: {}, prediction: {}".format(labels[idx][0], res))

        return count / len(images)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1.0 - np.tanh(x) ** 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def grad(self, X, y, idx):

        sample = np.asarray(X[idx]).reshape(-1, 1)

        # add bias in input layer
        sample = np.concatenate((sample, np.array([[1]])), axis=0).reshape(1, -1)

        # represent label as one-hot representation
        label = np.insert(np.zeros(self.O - 1), y[idx], [1]).reshape(self.O, 1)
        uj = np.dot(self.w1, sample.T).reshape(self.H, 1)
        uj = np.concatenate((uj, np.array([[1]])), axis=0)

        # add bias in hidden layer
        yj = self.tanh(uj).reshape(-1, 1)
        uk = np.dot(self.w2, yj).reshape(self.O, 1)
        zk = self.sigmoid(uk)

        # calculate gradient
        delta_k = (- label / zk + (1 - label) / (1 - zk)) * self.sigmoid_derivative(uk)
        grad_k = np.dot(delta_k, yj.T)
        delta_j = np.dot(self.w2.T, delta_k) * self.tanh_deriv(uj)
        grad_j = np.dot(delta_j[:self.H], sample)

        # see prediction correct or not
        correct = 0
        pred = zk.argmax(axis=0)
        if pred == y[idx]:
            correct = 1

        return grad_j, grad_k, correct


def preprocessing(X, mean, std):
    X -= mean
    X /= std
    return X


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("running %s" % ' '.join(sys.argv))

    X_train = np.load('official_data/train_image.npy').T.astype(np.float64)
    y_train = np.load('official_data/train_label.npy')
    X_test = np.load('official_data/test_image.npy').T.astype(np.float64)
    y_test = np.load('official_data/test_label.npy')

    means = np.mean(X_train, axis=0)
    s = np.std(X_train, axis=0, ddof=1)
    X_train = np.nan_to_num(preprocessing(X_train, means, s))
    X_test = np.nan_to_num(preprocessing(X_test, means, s))

    nn = NeuralNetwork(D=D, H=H, O=O, max_iter=60000)
    nn.trainNN(X_train, y_train)
    print("Test Accuracy: {}".format(nn.testNN(X_test, y_test)))
    print("Training Accuracy: {}".format(nn.acc[-1]))
    plt.plot(nn.acc)
    plt.title('Training accuracy vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()



