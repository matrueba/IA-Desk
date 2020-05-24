import numpy as np
from tensorflow.python.keras.datasets import boston_housing
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import h5py
import os


def load_raw(self):

    file = filedialog.askopenfilename(initialdir="./", title="Select Raw dataset",
                                          filetypes=(("csv files", "*.csv"), ("h5 files", "*.h5")))

    if file != None:
        path, extension = os.path.splitext(file)
        print("File: --" + path + "-- loaded succesfully")
        print("extension " + extension)

        if (extension == ".h5"):

            rawData = np.array(h5py.File(file))  # Read the file in HDF5 format
            self.TrainImagesLoaded = 0
            self.TestImagesLoaded = 0

        elif (extension == ".csv"):

            rawData = np.array(pd.read_csv(file, header=None))
            self.TrainImagesLoaded = 0
            self.TestImagesLoaded = 0

    return rawData


def load_train(self):

    file = filedialog.askopenfilename(initialdir="./", title="Select train file",
                                      filetypes=(("h5 files", "*.h5"), ("csv files", "*.csv")))

    if file != None:
        path, extension = os.path.splitext(file)
        print("File: --" + path + "-- loaded succesfully")
        print("extension " + extension)

        if (extension == ".h5"):

            train_dataset = h5py.File(file)  # Read the file in HDF5 format
            train_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
            train_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
            self.train_x = train_x_orig
            self.train_y = train_y_orig
            self.TrainImagesLoaded = 0

        elif (extension == ".csv"):

            train_dataset = pd.read_csv(file)
            train_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
            train_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
            self.train_x = train_x_orig
            self.train_y = train_y_orig
            self.TrainImagesLoaded = 0

        self.train_x = self.train_x.astype('float32')
        self.train_y = self.train_y.astype('float32')
        self.classes = len(np.unique(self.train_y))

        #This fragment of code is useful if some datasets have the rows and columns permuted
        #By default examples in rows and featurs in columns
        #
        #if messagebox.askyesno("Set distribution", "The training examples are distributed in rows?\n"
        #                                          "If not, it is assumed that the examples are distributed in columns "):

        #    print("Number of training examples: ", self.train_x.shape[1])
        #    print("Number of training parameters: ", self.train_x.shape[0])
        #else:
        #    self.train_x = np.transpose(self.train_x)
        #    self.train_y = np.transpose(self.train_y)
        #    print("Number of training examples: ", self.train_x.shape[0])
        #    print("Number of training parameters: ", self.train_x.shape[1])

        print("Number of label examples: ", self.train_y.shape[0])
        print("Classes: ", self.classes)
    else:
        print("File load error")


def load_images_train(self):

    file = filedialog.askopenfilename(initialdir="./", title="Select train file",
                                      filetypes=(("h5 files", "*.h5"), ("csv files", "*.csv")))


    if file != None:
        path, extension = os.path.splitext(file)
        print("File: --" + path + "-- loaded succesfully")
        print("extension " + extension)

        if (extension == ".h5"):

            train_dataset = h5py.File(file)  # Read the file in HDF5 format
            train_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
            train_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
            self.train_x = train_x_orig
            self.train_y = train_y_orig
            self.TrainImagesLoaded = 1

        elif (extension == ".csv"):

            train_dataset = pd.read_csv(file)
            train_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
            train_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
            self.train_x = train_x_orig
            self.train_y = train_y_orig
            self.TrainImagesLoaded = 1


        # Standardize data to have feature values between 0 and 1.
        self.train_x = self.train_x.astype('float32')
        self.train_y = self.train_y.astype('float32')
        self.train_x = self.train_x / 255.
        self.classes = len(np.unique(self.train_y))

        print("Number of training examples: ", self.train_x.shape[0])
        print("Size of images: "+str(self.train_x.shape[1])+"x"+str(self.train_x.shape[2]))
        print("Color channels of image: ", self.train_x.shape[3])
        print("Number of label examples: ",self.train_y.shape[0])
        print("Classes: ",  self.classes)

    else:
        print("File load error")


def load_test(self):

    file = filedialog.askopenfilename(initialdir="./", title="Select train file",
                                      filetypes=(("h5 files", "*.h5"), ("csv files", "*.csv")))

    if file != None:
        path, extension = os.path.splitext(file)
        print("File: --" + path + "-- loaded succesfully")
        print("extension " + extension)

        if (extension == ".h5"):

            test_dataset = h5py.File(file)  # Read the file in HDF5 format
            test_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
            test_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
            self.test_x = test_x_orig
            self.test_y = test_y_orig
            self.TestImagesLoaded = 0


        elif (extension == ".csv"):

            train_dataset = pd.read_csv(file)
            test_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
            test_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
            self.test_x = test_x_orig
            self.test_y = test_y_orig
            self.TestImagesLoaded = 0


        self.test_x = self.test_x.astype('float32')
        self.test_y = self.test_y.astype('float32')

        #This fragment of code is useful if some datasets have the rows and columns permuted
        #By default examples in rows and featurs in columns
        #
        #if messagebox.askyesno("Set distribution","The training examples are distributed in rows?\n"
        #                                          "If not, it is assumed that the examples are distributed in columns "):

        #    print("Number of test examples: ", self.test_x.shape[1])
        #    print("Number of test parameters: ", self.test_x.shape[0])
        #else:
        #    self.test_x = np.transpose(self.test_x)
        #    self.test_y = np.transpose(self.test_y)
        #    print("Number of test examples: ", self.test_x.shape[0])
        #    print("Number of test parameters: ", self.test_x.shape[1])

        print("Number of label examples: ", self.test_y.shape[0])

    else:
        print("File load error")


def load_images_test(self):

    file = filedialog.askopenfilename(initialdir="./", title="Select train file",
                                      filetypes=(("h5 files", "*.h5"), ("csv files", "*.csv")))

    if file != None:
        path, extension = os.path.splitext(file)
        print("File: --" + path + "-- loaded succesfully")
        print("extension " + extension)


        if (extension == ".h5"):

            test_dataset = h5py.File(file)  # Read the file in HDF5 format
            test_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
            test_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
            self.test_x = test_x_orig
            self.test_y = test_y_orig
            self.TestImagesLoaded = 1

        elif (extension == ".csv"):

            train_dataset = pd.read_csv(file)
            test_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
            test_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
            self.test_x = test_x_orig
            self.test_y = test_y_orig
            self.TestImagesLoaded = 1

        # Standardize data to have feature values between 0 and 1.
        self.test_x = self.test_x.astype('float32')
        self.test_y = self.test_y.astype('float32')
        self.test_x = self.test_x / 255.

        print("Number of test examples: ", self.test_x.shape[0])
        print("Size of images: " + str(self.test_x.shape[1]) + "x" + str(self.test_x.shape[2]))
        print("Color channels of image: ", self.test_x.shape[3])
        print("Number of label examples: ", self.test_y.shape[0])

    else:
        print("File load error")


def load_internal_datasets(self):
    #Load internal keras or tensorflow datasets as cifar10, boston_housing...
    (self.train_x, self.train_y),(self.test_x, self.test_y) = boston_housing.load_data()

    print("Boston Housing dataset loaded")
    print("Number of train examples: ", self.train_x.shape[0])
    print("Number of test examples: ", self.test_x.shape[0])
    print("Number of train parameters: ", self.train_x.shape[1])
    print("Number of test parameters: ", self.test_x.shape[1])
  



def loadPretrained(self):
    print("Loading pretrained_network")
    file = filedialog.askopenfilename(initialdir="./", title="Select file",
                                      filetypes=(("numpy files", "*.npy"), ("pikle files", "*.pkl")))

    if file != None:
        path, extension = os.path.splitext(file)
        print("File: --" + path + "-- loaded succesfully")
        print("extension " + extension)
        self.params = np.load(file)
        print(self.params)
        #TODO

    else:
        print("File load error")