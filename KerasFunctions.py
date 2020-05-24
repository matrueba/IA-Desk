import tensorflow as tf
import numpy as np
from tensorflow.python.keras import  initializers
from tensorflow.python.keras.layers import  Dense, Activation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import Callback


def convert_to_one_hot(Y_train, Y_test, classes):
    Y_train = to_categorical(Y_train, classes)
    Y_test = to_categorical(Y_test, classes)
    return Y_train, Y_test

def select_optimizer(dictionary):
    learning_rate = dictionary['lambda']
    decay = dictionary['decay']
    momentum = dictionary['momentum']
    RMSmomentum = dictionary['RMSmomentum']
    beta1 = dictionary['beta1']
    beta2 = dictionary['beta2']
    AdamEpsilon = dictionary['AdamEpsilon']
    RMSEpsilon = dictionary['RMSEpsilon']
    optimizer = None
    if dictionary['type'] == "adam":
        optimizer = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=AdamEpsilon)
    elif dictionary['type'] == "GD":
        optimizer = SGD(lr=learning_rate)
    elif dictionary['type'] == "RMS":
        optimizer = RMSprop(lr=learning_rate, epsilon=RMSEpsilon, decay=decay )#tambien tiene el parametro rho
    elif dictionary['type'] == "momentum":
        optimizer = Nadam(lr=learning_rate, beta_1=beta1, beta_2=beta2)
    return optimizer


class printMetricsOnScren(Callback):

    def __init__(self, controller):
        self.losses = []
        self.acc = []
        self.controller = controller

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        if  epoch % 100 == 0:
            print("Cost after epoch %i: %f" % (epoch, logs.get('loss')))
            self.controller.valCost.set("Cost after epoch %i: %f" % (epoch, logs.get('loss')))
            self.controller.valAcc.set("Accuracy after epoch %i: %f" % (epoch, logs.get('acc')))
            self.controller.master.update()

def nn_model(classes, layer_dims, nn_dims, m, n_w, optimizer):

    loss = 'categorical_crossentropy'
    initializer = 'glorot_uniform'

    model = Sequential()
    for l in range(0,layer_dims):
        if (l==0):
            model.add(Dense(nn_dims[l], input_shape=(n_w, ), kernel_initializer=initializer))
            model.add(Activation('relu'))
        else:
            model.add(Dense(nn_dims[l]))
            model.add(Activation('relu'))
    if (classes > 1):
        model.add(Dense(classes))
        model.add(Activation('softmax'))
    else:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def linear_reression_model(dims, learning_rate):
    model = Sequential()
    model.add(Dense(1, input_dim=dims, activation='linear'))
    optimizer = SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model
