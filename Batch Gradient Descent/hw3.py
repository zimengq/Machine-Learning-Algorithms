#!~/anaconda3/bin/ python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
18661 Introduction to Machine Learning for Engineers HW3 Q3
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("running %s" % ' '.join(sys.argv))


class BatchGradientDescent(object):
    """
    Batch Gradient Descent
    @:param step: setp when iteration
    @:param bound: lower bound of sigmoid function
    """
    def __init__(self, step, bound):
        self.step = step
        self.bound = bound
        self.w = None
        self.b = None

    """
    Calculate sigmoid output
    """
    def sigmoid(self, x):
        if x < 0:
            ans = np.exp(x) / (np.exp(x) + 1)
        else:
            ans = 1 / (1 + np.exp(-x))
        if ans > 1 - self.bound:
            ans = 1 - self.bound
        elif ans < self.bound:
            ans = self.bound
        return ans

    """
    Calculate cross entropy
    """
    def cross_entropy(self, X, y, w, b, lambda_):
        e = lambda_ * (np.linalg.norm(w.transpose()) ** 2)
        for i in range(X.shape[0]):
            e -= (y[i] * np.log(self.sigmoid(b + np.dot(w.transpose(), X[i])))
                  + (1 - y[i]) * np.log(1 - self.sigmoid(b + np.dot(w.transpose(), X[i]))))
        return e

    """
    Calculate gradient of w
    """
    def grad_w(self, X, y, w, b, lambda_):
        n = X.shape[0]
        d = X.shape[1]
        grad = np.zeros(d)
        for i in range(n):
            h = self.sigmoid(b + np.dot(w.transpose(), X[i]))
            grad += (h - y[i]) * X[i]
        grad += 2 * lambda_ * w
        return grad

    """
    Calculate gradient of b
    """
    def grad_b(self, X, y, w, b):
        n = X.shape[0]
        d = X.shape[1]
        grad = 0.0
        for i in range(n):
            grad += (self.sigmoid(b + np.dot(w.transpose(), X[i])) - y[i])
        return grad

    """
    Fit training data
    @:param lambda_ = 0 means no regularization
    """
    def fit(self, X, y, iter_=50, lambda_=0.1):
        if not isinstance(X, np.ndarray):
            raise ValueError("Wrong type. Require numpy.ndarray, found %s " % type(X))
        n = X.shape[0]
        d = X.shape[1]
        self.w = np.zeros(d)
        self.b = 0.1

        cross_entropy_record = []
        epoch = 0
        while epoch < iter_:
            entropy = self.cross_entropy(X, y, self.w, self.b, lambda_)
            w_grad = self.grad_w(X, y, self.w, self.b, lambda_)
            b_grad = self.grad_b(X, y, self.w, self.b)
            self.w -= self.step * w_grad
            self.b -= self.step * b_grad
            epoch += 1
            logger.info("Step: {}, Cross-entropy: {}".format(epoch, entropy))

            cross_entropy_record.append(entropy)

        return cross_entropy_record, self.w


def load_dict(dict_file):
    with open(dict_file) as df:
        vocab = df.read().split("\n")
    return vocab


def read_txt(txt_file):
    text = ""
    with open(txt_file) as tf:
        try:
            text = tf.read()
        except IOError:
            logger.info("Can not open file {}".format(txt_file))
        except UnicodeDecodeError:
            logger.info("Can not decode file {}".format(txt_file))
    return text


def load_text(text_path):
    walk = os.walk(os.path.join(dir_path, text_path))
    docs = []

    for root, dirs, files in walk:
        for file in files:
            if file.endswith(".txt") and (not(file.startswith("._"))):
                fp = os.path.join(root, file)
                # logger.info("Loading document {}".format(fp))
                docs.append(read_txt(fp))
    return docs


def tokenize(text):
    """
    I think all punctuations plus "\n" should be removed.
    :param text: raw text
    :return: tokens
    """
    delimiters = [" ", ".", ",", "?", "\\n"
                  # "~", "`", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "-", "+", "="
                  # "{", "}", "[", "]", "|", "\\", ":", ";", "\"", "\'", "<", ">", "/"
                  ]
    for delimiter in delimiters:
        text = text.replace(delimiter, " ")
    text = text.split()
    words = [word.lower() for word in text]
    return words


def generate_bow(docs, vocab):
    bow_vector = np.zeros((len(docs), len(vocab)))
    for index, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                bow_vector[index, vocab.index(token)] += 1
    return bow_vector


if __name__ == '__main__':
    # Load train and test data
    vocab = load_dict(os.path.join(dir_path, "hw3_data/spam/dic.dat"))
    train_spam = load_text("hw3_data/spam/train/spam")
    train_ham = load_text("hw3_data/spam/train/ham")
    test_spam = load_text("hw3_data/spam/test/spam")
    test_ham = load_text("hw3_data/spam/test/ham")

    # Tokenization
    # This step could also be donw while loading text
    train_spam = [[tokenize(doc), 1] for doc in train_spam]
    train_ham = [[tokenize(doc), 0] for doc in train_ham]
    test_spam = [[tokenize(doc), 1] for doc in test_spam]
    test_ham = [[tokenize(doc), 0] for doc in test_ham]

    X_train = [data[0] for data in train_spam + train_ham]
    X_test = [data[0] for data in test_spam + test_ham]
    y_train = [data[1] for data in train_spam + train_ham]
    y_test = [data[1] for data in test_spam + test_ham]

    X_train = generate_bow(X_train, vocab)
    train_vocab = {}
    for term, occurrence in zip(vocab, X_train.sum(axis=0)):
        train_vocab[term] = occurrence
    train_vocab = sorted(train_vocab.items(), key=lambda x: x[1], reverse=True)
    print("Top 3 keywords %s" % train_vocab[0:3])

    etas = [0.001, 0.01, 0.05, 0.1, 0.5]
    steps = np.linspace(1, 50, 50)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Cross-entropy Cruve Versus Step (without regularization)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy")
    for eta in etas:
        logger.info("Eta=%s, Lambda=0" % eta)
        clf = BatchGradientDescent(step=eta, bound=1e-16)
        entropy, weight = clf.fit(X_train, y_train, lambda_=0)
        ax.plot(steps, entropy)
        logger.info("L2 norm: {}".format(np.linalg.norm(weight)))
    ax.legend(["eta=0.001", "eta=0.01", "eta=0.05", "eta=0.1", "eta=0.5"])

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Cross-entropy Cruve Versus Step (lambda=0.1)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cross-entropy")
    for eta in etas:
        logger.info("Eta=%s, Lambda=0.1" % eta)
        clf = BatchGradientDescent(step=eta, bound=1e-16)
        entropy, weight = clf.fit(X_train, y_train, lambda_=0.1)
        ax2.plot(steps, entropy)
    ax2.legend(["eta=0.001", "eta=0.01", "eta=0.05", "eta=0.1", "eta=0.5"])

    for lambda_ in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        logger.info("Eta=0.01, Lambda=%s" % lambda_)
        clf = BatchGradientDescent(step=0.01, bound=1e-16)
        entropy, weight = clf.fit(X_train, y_train, lambda_=lambda_)
        logger.info("L2 norm: {}".format(np.linalg.norm(weight)))

    while True:
        try:
            plt.show()
        except UnicodeDecodeError:
            continue
        break






