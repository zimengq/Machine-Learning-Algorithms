#!~/anaconda3/bin/ python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
18661 Introduction to Machine Learning for Engineers HW4 Q5
Implementing a SVM
"""

import os
import sys
import logging
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

dir_path = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("running %s" % ' '.join(sys.argv))


class SVM(object):

    def __init__(self, C):
        self.w = None
        self.b = None
        self.C = C

    def get_params(self, deep=False):
        return {'C': self.C}

    def fit(self, X, y):
        """
        Train linear SVM (primal form)

        Argument:
            X: train data, N*D matrix, each row as a sample and each column as a feature
            y: train label, N*1 vector, each row as a label
            C: tradeoff parameter (on slack variable side) (packaged in class SVM, self.C)

        Return:
            w: feature vector (column vector)
            b: bias term
        """

        # Initializing values and computing H
        m, n = X.shape
        y = y.reshape(-1, 1) * 1.
        X_dash = y * X
        H = np.dot(X_dash, X_dash.T) * 1.

        # Converting into cvxopt format - as previously
        P = matrix(H)
        q = matrix(-np.ones((m, 1)))
        G = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False

        # Run solver
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        # w parameter in vectorized form
        self.w = ((y * alphas).T @ X).reshape(-1, 1)

        # Selecting the set of indices S corresponding to non zero parameters
        S = (alphas > 1e-4).flatten()

        # Computing b
        self.b = y - np.dot(X, self.w)

        return self.w, self.b

    def test_svm(self, X, y):
        """
        Test linear SVM

        Argument:
            X: test data, M*D matrix, each row as a sample and each column as a feature
            y: test label, M*1 vector, each row as a label
            w: feature vector (packaged in class SVM, self.w)
            b: bias term (packaged in class SVM, self.b)

        Return:
            test_accuracy: a float between [0, 1] representing the test accuracy
        """

        correct = 0
        pred = y * (np.dot(X, self.w) + self.b[0])
        for n in range(len(pred)):
            if pred[n] > 0:
                correct += 1

        return correct / len(X)

    def signfunc(self, inp):
        """
        Argument:
            inp: input number
        :return: 1 if inp > 0, 0 if inp = 0, else -1
        """

        if inp > 0:
            return 1
        elif inp == 0:
            return 0
        else:
            return -1


def load_data(data_file):
    with open(data_file) as f:
        data = f.readlines()
    for i in range(len(data)):
        try:
            data[i] = [float(x) for x in data[i].split(' ')]
        except ValueError:
            print("error", "on line", i)
    return np.array(data)


def preprocessing(X):
    means = np.mean(X, axis=0)
    s = np.std(X, axis=0, ddof=1)
    X = (X - means) / s
    return X


def scorer(estimator, X, y):
    correct = 0
    pred = y * (np.dot(X, estimator.w) + estimator.b[0])
    for n in range(len(pred)):
        if pred[n] > 0:
            correct += 1

    return correct / len(X)


if __name__ == '__main__':
    X_train = load_data('train_data.txt')
    y_train = load_data('train_label.txt')
    X_test = load_data('test_data.txt')
    y_test = load_data('test_label.txt')

    X_train = preprocessing(X_train)

    # cross validation for tuning on C
    for power in range(-6, 7):
        C = 4 ** power
        logger.info("C={}, ".format(C))
        svm = SVM(C=C)
        scores = cross_validate(svm, X_train, y_train, scoring=scorer, cv=5, return_train_score=False)
        logger.info("test data: {}".format(scores))
        logger.info("average accuracy: {}".format(scores['test_score'].mean()))
        logger.info("average training time: {}".format(scores['fit_time'].mean()))

    # test using optimal C
    svm = SVM(C=4 ** -5)
    svm.fit(X_train, y_train)
    print("test accuracy: {}".format(svm.test_svm(X_test, y_test)))