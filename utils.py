#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:30:42 2022

@author: adowa
"""

import tensorflow as tf
from tensorflow.keras import layers,models,initializers,optimizers,losses,metrics
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

def build_logistic_regression(
                    input_size              = 100,
                    output_size             = 2,
                    special                 = False,
                    kernel_regularizer      = None,
                    activity_regularizer    = None,
                    print_model             = False,
                    ):
    """
    This function builds a logistic regression classifier
    
    Inputs
    ---
    input_size: int, the 2nd dimension of the input data
    output_size: int, the number of classes
    special: bool, just in case I want to use this special combination of activation functions
    kernel_regularizer: None or tf.keras.regularizers, to control for the layer weights
    activity_regularizer: None or tf.keras.regularizers, to control for the output sparsity
    print_model: bool, whether to show the model architecture in a summary table
    
    Outputs
    ---
    logistic_regression: tf.keras.models.Model, a tf-keras model that has .fit()
    """
    tf.random.set_seed(12345)
    input_layer = layers.Input(shape        = (input_size,),
                               name         = "input_layer",)
    if special:
        middle_layer = layers.Dense(
                                      units                 = output_size,
                                      activation            = 'selu',
                                      use_bias              = True,
                                      kernel_initializer    = initializers.LecunNormal(),
                                      kernel_regularizer    = kernel_regularizer,
                                      activity_regularizer  = activity_regularizer,
                                      name                  = 'middle_layer',
                                      )(input_layer)
        logistic_layer = layers.Activation('softmax',
                                           name = 'logistic_layer')(middle_layer)
    else:
        logistic_layer = layers.Dense(units                 = output_size,
                                      activation            = 'softmax',
                                      use_bias              = True,
                                      kernel_initializer    = initializers.HeNormal(),
                                      kernel_regularizer    = kernel_regularizer,
                                      activity_regularizer  = activity_regularizer,
                                      name                  = 'logistic_layer'
                                      )(input_layer)
    logistic_regression = models.Model(input_layer,
                                       logistic_layer,
                                       name = 'logistic_regression')
    if print_model:
        print(logistic_regression.summary())
    # don't forget to complile the model once you build it
    return logistic_regression

# the most important helper function: early stopping and model saving
def make_CallBackList(model_name,monitor='val_loss',mode='min',verbose=0,min_delta=1e-4,patience=50,frequency = 1):
    
    """
    Make call back function lists for the keras models
    
    Parameters
    -------------------------
    model_name : str,
        directory of where we want to save the model and its name
    monitor : str, default = 'val_loss'
        the criterion we used for saving or stopping the model
    mode : str, default = 'min'
        min --> lower the better, max --> higher the better
    verboser : int or bool, default = 0
        printout the monitoring messages
    min_delta : float, default = 1e-4
        minimum change for early stopping
    patience : int, default = 50
        temporal windows of the minimum change monitoring
    frequency : int, default = 1
        temporal window steps of the minimum change monitoring
    
    Return
    --------------------------
    CheckPoint : tensorflow.keras.callbacks
        saving the best model
    EarlyStopping : tensorflow.keras.callbacks
        early stoppi
    """
    checkPoint = ModelCheckpoint(model_name,# saving path
                                 monitor          = monitor,# saving criterion
                                 save_best_only   = True,# save only the best model
                                 mode             = mode,# saving criterion
                                 verbose          = verbose,# print out (>1) or not (0)
                                 )
    earlyStop = EarlyStopping(   monitor          = monitor,
                                 min_delta        = min_delta,
                                 patience         = patience,
                                 verbose          = verbose, 
                                 mode             = mode,
                                 )
    return [checkPoint,earlyStop]

def compile_logistic_regression(
                    model,
                    model_name      = 'temp.h5',
                    optimizer       = None,
                    loss_function   = None,
                    metric          = None,
                    callbacks       = None,
                    learning_rate   = 1e-2,
                    tol             = 1e-4,
                    patience        = 5,
                    ):
    """
    Inputs
    ---
    model: tf.keras.models.Model or callable tf objects
    model_name: str, directory of where we want to save the model and its name
    optimizer: None or tf.keras.optimizers, default = SGD
    loss_function: None or tf.keras.losses, default = BinaryCrossentropy
    metric: None or tf.keras.metrics, default = AUC
    callbacks: None or list of tf.keras.callbacks, default = [checkpoint,earlystopping]
    learning_rate: float, learning rate, default = 1e-2,
    tol: float, for determining when to stop training, default = 1e-4,
    patience: int, for determing when to stop training, default = 5,
    
    Outputs
    ---
    model: tf.keras.models.Model or callable tf objects
    callbacks:ist of tf.keras.callbacks
    """
    if optimizer is None:
        optimizer       = optimizers.SGD(learning_rate = learning_rate,)
    if loss_function is None:
        loss_function   = losses.BinaryCrossentropy()
    if metric is None:
        metric          = metrics.AUC()
    if callbacks is None:
        callbacks       = make_CallBackList(
                                      model_name    = model_name,
                                      monitor       = 'val_loss',
                                      mode          = 'min',
                                      verbose       = 0,
                                      min_delta     = tol,
                                      patience      = patience,
                                      frequency     = 1,
                                      )
    model.compile(optimizer = optimizer,
                  loss      = loss_function,
                  metrics   = [metric],
                  )
    return model,callbacks