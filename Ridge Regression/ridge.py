# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import numpy as np
import matplotlib.pyplot as plt
import argparse
import numbers
import math
import os
import sys
import logging

from operator import itemgetter


class RidgeRegression(object):
    """
     Ridge Regression(Linear Least Squares Regression with Tikhonov regularization).
    """

    def __init__(self):
        self.beta = None

    def fit(self, X, y, alpha=0):
        """
        Fits the ridge regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: response variable vector for n examples
        alpha: regularization parameter.
        """
        if not isinstance(alpha, numbers.Number) or alpha < 0:
            raise ValueError("Penalty term must be positive; got (alpha=%r)"
                             % alpha)

        X_trans = np.matrix.transpose(X)
        I = np.identity(X.shape[1])
        self.beta = np.dot(np.linalg.inv((np.dot(X_trans, X) + alpha * I)), np.dot(X_trans, y))

    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nxp matrix of n examples with p covariates

        Returns
        ----------
        response variable vector for n examples
        """
        if self.beta is None:
            raise SystemExit("Train model before predict!")

        return np.dot(X, self.beta)

    def validate(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.

        Arguments
        ----------
        X: nxp matrix of n examples with p covariates
        y: response variable vector for n examples

        Returns
        ----------
        RMSE when model is used to predict y
        """
        return np.sqrt(((self.predict(X) - y) ** 2).mean())


# run command:
# python ridge.py --X_train_set=Xtraining.csv --y_train_set=Ytraining.csv --X_val_set=Xvalidation.csv --y_val_set=Yvalidation.csv --y_test_set=Ytesting.csv --X_test_set=Xtesting.csv

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s " % ' '.join(sys.argv))

    # Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Ridge Regression Model')
    parser.add_argument('--X_train_set', required=True,
                        help='The file which contains the covariates of the training dataset.')
    parser.add_argument('--y_train_set', required=True,
                        help='The file which contains the response of the training dataset.')
    parser.add_argument('--X_val_set', required=True,
                        help='The file which contains the covariates of the validation dataset.')
    parser.add_argument('--y_val_set', required=True,
                        help='The file which contains the response of the validation dataset.')
    parser.add_argument('--X_test_set', required=True,
                        help='The file which containts the covariates of the testing dataset.')
    parser.add_argument('--y_test_set', required=True,
                        help='The file which containts the response of the testing dataset.')

    args = parser.parse_args()

    # Parse training dataset
    X_train = np.genfromtxt(args.X_train_set, delimiter=',')
    y_train = np.genfromtxt(args.y_train_set, delimiter=',')

    # Parse validation set
    X_val = np.genfromtxt(args.X_val_set, delimiter=',')
    y_val = np.genfromtxt(args.y_val_set, delimiter=',')

    # Parse testing set
    X_test = np.genfromtxt(args.X_test_set, delimiter=',')
    y_test = np.genfromtxt(args.y_test_set, delimiter=',')

    # find the best regularization parameter
    lambda_rec = []
    rmse_rec = []
    predicted_rec = []
    coef_rec = []
    model = RidgeRegression()
    for _lambda in range(1, 2000, 2):
        _lambda /= 1000
        model.fit(X_train, y_train, alpha=_lambda)
        predict_result = model.predict(X_test)
        rmse = model.validate(X_val, y_val)
        lambda_rec.append(_lambda)
        rmse_rec.append(rmse)
        predicted_rec.append(predict_result)
        coef_rec.append(model.beta)
        # print("Lambda: {}  RMSE: {}".format(_lambda, rmse))
    optimal_lambda = lambda_rec[rmse_rec.index(min(rmse_rec))]
    optimal_predicted = predicted_rec[rmse_rec.index(min(rmse_rec))]
    logger.info("Optimal Lambda: {}".format(optimal_lambda))
    logger.info("Minimum RMSE: {}".format(min(rmse_rec)))
    
    # plot rmse versus lambda
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambda_rec, rmse_rec)
    ax.set_title("RMSE Cruve Versus Lambda")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("RMSE")
    ax.axvline(optimal_lambda, c = "r", lw = 2)
    ax.set_ylim(0, 20)
    ax.annotate(
        'Lowest RMSE, Lambda={}'.format(optimal_lambda), 
        xy=(optimal_lambda, min(rmse_rec)), 
        # xytext=(3, 1.5),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )
    
    #  plot predicted versus real value
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    pre_vs_real = sorted(zip(y_test, optimal_predicted), key=lambda x:x[0])
    real_value = [_[0] for _ in pre_vs_real]
    predicted = [_[1] for _ in pre_vs_real]
    ax2.plot(real_value, predicted)
    ax2.set_title("predicted Cruve Versus Real Value")
    ax2.set_xlabel("Real Value")
    ax2.set_ylabel("Predicted Value")
    
    #  plot regression coefficients
    labels = [
        "beta0",
        "beta1",
        "beta2",
        "beta3",
        "beta4",
        "beta5",
        "beta6",
        "beta7",
        "beta8",
        "beta9",
        "beta10",
        "beta11",
        "beta12",
        "beta13",
        "beta14",
        "beta15",
        "beta16",
        "beta17",
        "beta18",
        "beta19"
    ]
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    for i in range(0,20):
        coef = [_[i] for _ in coef_rec]
        ax3.plot(lambda_rec, coef)
    ax3.set_title("Beta Curve Versue Lambda")
    ax3.set_xlabel("Lambda")
    ax3.set_ylabel("Elements of Beta")
    ax3.legend(labels)

    while True:
        try:
            plt.show()
        except UnicodeDecodeError:
            continue
        break