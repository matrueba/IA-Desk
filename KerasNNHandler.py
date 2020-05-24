import math
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
from KerasFunctions import *
import matplotlib
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import json
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping


def plot_graph_keras(self, costs, accuracy):

    plot_frame = Frame(self, bd=3, relief=SUNKEN)
    plot_frame.grid(column=0, row=0, columnspan=2)

    fig = Figure(figsize=(5, 2.5), dpi=100)
    a = fig.add_subplot(111)
    a.plot(np.squeeze(costs))
    a.plot(np.squeeze(accuracy))

    #DrawingArea
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

    def on_key_event(event):
        print('you pressed %s' % event.key)
        key_press_handler(event, canvas, toolbar)

    canvas.mpl_connect('key_press_event', on_key_event)


def save_Parameters_keras(model):

    path = filedialog.asksaveasfilename(initialdir="C:/Datasheets_RNN/Trained_models",
                                        title="Save trained NN, don't forget put the right extension",
                                        filetypes=(("HDF5 file", "*.h5"), ("all files", "*.")))

    if path != None:
        print("Se guardará en la ruta " + path)
        model.save_weights(path)
        print("Parametros guardados con exito")

    else:

        print("Error al guardar el archivo")

def save_Model_keras(model):

    path = filedialog.asksaveasfilename(initialdir="C:/Datasheets_RNN/Trained_models",
                                        title="Save trained NN, don't forget put the right extension",
                                        filetypes=(("HDF5 file", "*.h5"), ("all files", "*.")))

    if path != None:
        print("Se guardará en la ruta " + path)
        model.save(path)
        print("Modelo guardado con exito")

    else:

        print("Error al guardar el archivo")

"""
###################################################################################################
#   Implements a n-layer tensorflow neural network.                                               #
#                                                                                                 #
#   Arguments:                                                                                    #
#    X_train -- training set, of shape                                                            #
#    Y_train -- test set, of shape                                                                #
#    X_test -- training set, of shape                                                             #
#    Y_test -- test set, of shape                                                                 #
#    dictionary who contains                                                                      #
#    learning_rate -- learning rate of the optimization                                           #
#    num_epochs -- number of epochs of the optimization loop                                      #
#    minibatch_size -- size of a minibatch                                                        #
#    print_cost -- True to print the cost every 100 epochs                                        #
#                                                                                                 #
#   Returns:                                                                                      #
#    parameters -- parameters learnt by the model. They can then be used to predict.              #
#                                                                                                 #
###################################################################################################
"""

def model_train(self, X_train, Y_train, X_test, Y_test, classes, dictionary):
    nn_dims = dictionary['dims']
    num_layers = len(nn_dims)
    num_epochs = dictionary['epochs']
    minibatch_size = dictionary['batch']

    (m, n_w) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    optimizer = select_optimizer(dictionary=dictionary)
    model = nn_model(classes=classes, layer_dims=num_layers, nn_dims=nn_dims, m=m,  n_w=n_w, optimizer=optimizer)
    model.summary()

    #checkpoint = ModelCheckpoint('/data', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint = ModelCheckpoint(filepath="", save_best_only=True)
    callback_list = [printMetricsOnScren(self)]

    #This code avoid to_categorical error if the classes dont start at 0
    min_class = min(Y_train)

    Y_train = to_categorical(Y_train - min_class, classes)
    Y_test = to_categorical(Y_test - min_class, classes)

    history = model.fit(X_train, Y_train, batch_size=minibatch_size, epochs=num_epochs, verbose=2, callbacks=callback_list)
    test_score, test_accuracy = model.evaluate(X_test, Y_test)

    return history, test_score, test_accuracy, model


def linear_regresion_train(self, X_train, Y_train, X_test, Y_test, dictionary):
    dims = dictionary['dims']
    n_epochs = dictionary['epochs']
    learning_rate = dictionary['lambda']
    model_lr = linear_reression_model(dims=dims, learning_rate=learning_rate)
    model_lr.fit(X_train, Y_train, batch_size=1, epochs=n_epochs, shuffle=False)

