import math
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
from TFfunctions import *
import matplotlib
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import json


def plot_graph(self, costs):

    plot_frame = Frame(self, bd=3, relief=SUNKEN)
    plot_frame.grid(column=0, row=0, columnspan=2)

    fig = Figure(figsize=(5, 2.5), dpi=100)
    a = fig.add_subplot(111)
    a.plot(np.squeeze(costs))

    #DrawingArea
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    canvas.show()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

    def on_key_event(event):
        print('you pressed %s' % event.key)
        key_press_handler(event, canvas, toolbar)

    canvas.mpl_connect('key_press_event', on_key_event)


def save_Parameters(params):

    path = filedialog.asksaveasfilename(initialdir="./",
                                        title="Save trained NN, don't forget put the right extension",
                                        filetypes=(("numpy file", "*.npy"), ("all files", "*.")))

    if path != None:

        print("The following path will be saved " + path)

        np.save(path, params)
        print("Parameters of network saved succesfully")

    else:

        print("File load error")


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

def nn_model(self, X_train, Y_train, X_test, Y_test, dictionary, print_cost=True):

    layer_dims = dictionary['dims']
    num_epochs = dictionary['epochs']
    minibatch_size  = dictionary['batch']
    learning_rate = dictionary['lambda']
    decay = dictionary['decay']
    momentum = dictionary['momentum']
    RMSmomentum = dictionary['RMSmomentum']
    beta1 = dictionary['beta1']
    beta2 = dictionary['beta2']
    AdamEpsilon= dictionary['AdamEpsilon']
    RMSEpsilon = dictionary['RMSEpsilon']


    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.

    if dictionary['type'] == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=AdamEpsilon,).minimize(cost)
    elif dictionary['type'] == "GD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    elif dictionary['type'] == "RMS":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=RMSmomentum, epsilon=RMSEpsilon).minimize(cost)
    elif dictionary['type'] == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(cost)


    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                self.var.set("Cost after epoch %i: %f" % (epoch, epoch_cost))
                self.master.update()

            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:" + str(train_accuracy))
        print("Test Accuracy:" + str(test_accuracy))

        return costs, parameters, train_accuracy, test_accuracy





