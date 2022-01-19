#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:30:56 2022

@author: adowa
"""
import numpy as np
import tensorflow as tf
from utils import (build_logistic_regression,
                   compile_logistic_regression)
from tensorflow.keras import regularizers
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    n_epochs = int(1e3) # just a large number
    print_train = True
    # clear memory states
    tf.keras.backend.clear_session()
    # generate random test data
    X,y = make_classification(n_samples             = 150,
                              n_features            = 100,
                              n_informative         = 3,
                              n_redundant           = 10,
                              n_classes             = 2,
                              n_clusters_per_class  = 4,
                              flip_y                = .01,
                              class_sep             = .75,# how easy to separate the two classes
                              shuffle               = True,
                              random_state          = 12345,
                              )
    # one-hot encoding for softmax
    y = y.reshape((-1,1))
    y = np.hstack([y,1-y])
    # split the data into train, validation, and test
    X_train,X_test,y_train,y_test   = train_test_split(X,y,test_size = .1,random_state = 12345)
    X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size = .1,random_state = 12345)
    # add some 0.5 labeled data - don't use too much
    X_noise = np.random.normal(X_train.mean(),X_train.std(),size = (int(X_train.shape[0]/4),100))
    y_noise = np.array([[0.5,0.5]] * int(X_train.shape[0]/4))
    X_train = np.concatenate([X_train,X_noise])
    y_train = np.concatenate([y_train,y_noise])
    
    # X_noise = np.random.normal(X_test.mean(),X_test.std(),size = (int(X_test.shape[0]/2),100))
    # y_noise = np.array([[0.5,0.5]] * int(X_test.shape[0]/2))
    # X_test  = np.concatenate([X_test,X_noise])
    # y_test  = np.concatenate([y_test,y_noise])
    # build the model
    tf.random.set_seed(12345)
    logistic_regression = build_logistic_regression(
                            input_size              = X_train.shape[1],
                            output_size             = 2,
                            special                 = False,
                            kernel_regularizer      = regularizers.L2(l2 = 1e-3),
                            activity_regularizer    = regularizers.L1(l1 = 1e-3),
                            print_model             = True,
                            )
    # compile the model
    logistic_regression,callbacks = compile_logistic_regression(
                                    logistic_regression,
                                    model_name      = 'temp.h5',
                                    optimizer       = None,
                                    loss_function   = None,
                                    metric          = None,
                                    callbacks       = None,
                                    learning_rate   = 1e-3,
                                    tol             = 1e-4,
                                    patience        = 10,
                                    )
    # train and validate the model
    logistic_regression.fit(
                            X_train,
                            y_train,
                            batch_size      = 4,
                            epochs          = n_epochs,
                            verbose         = print_train,
                            callbacks       = callbacks,
                            validation_data = (X_valid,y_valid),
                            shuffle         = True,
                            class_weight    = None,# tf has this but I don't think it is the same as sklearn
                            )
    y_pred = logistic_regression.predict(X_test)
    print(roc_auc_score(y_test,y_pred,))