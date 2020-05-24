from tkinter import filedialog, ttk
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from NNhandler import *
from tensorflow.python.framework import ops
from TFfunctions import *
from KerasNNHandler import *
from AuxWindows import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
import os
from tensorflow.python.keras.models import load_model

#This frame is temporal, only for test
class Tabs(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.notebook = ttk.Notebook(self, height=372, width=594)
        self.notebook.pack()

        self.Label1 = ttk.Label(self.notebook, text='tab1')
        self.Label2 = ttk.Label(self.notebook, text='tab2')

        self.notebook.add(self.Label1, text='tab1')
        self.notebook.add(self.Label2, text='tab2')

"""----------------------------------------------------------------------------------------------------------------"""
def Return(controller):
    print("Volviendo a pantalla principal")
    controller.show_frame(StartWindow)

def RefreshTrain(self, controller):
    plotRandomExampleTrain(self, controller)


def plotRandomExampleTrain(self, controller):

    plot_frame = Frame(self, bd=3, relief=SUNKEN)
    plot_frame.grid(column=0, row=2, columnspan=3)

    fig = Figure(figsize=(5, 2.5), dpi=80)
    a = fig.add_subplot(111)
    idx = np.arange(controller.train_x.shape[0])
    choice = np.random.choice(idx, replace=False)
    a.imshow(controller.train_x[choice])

    #DrawingArea
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)


class InfoTrainWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.trainingExamples = 0
        self.trainingParameters = 0
        self.trainingClasses = 0
        labelNumberInfo = Label(self, text="Number of train examples", font=("Helvetica", 10))
        labelNumberInfo.grid(row=0, column=0, pady=(10, 5), padx=10)
        labelParametersInfo = Label(self, text="Number of train parameters", font=("Helvetica", 10))
        labelParametersInfo.grid(row=0, column=1, pady=(10, 5), padx=10)
        labelClassesInfo = Label(self, text="Number of classes", font=("Helvetica", 10))
        labelClassesInfo.grid(row=0, column=2, pady=(10, 5), padx=10)
        buttonReturn = Button(self, text="Return", command=lambda: Return(self.controller), font=("Agency FB", 14), width=16)
        buttonReturn.grid(row=4, column=0, pady=25, columnspan=2)
        buttonRefresh = Button(self, text="Refresh", command=lambda: RefreshTrain(self, self.controller),font=("Agency FB", 14), width=16)
        buttonRefresh.grid(row=4, column=1, pady=25, columnspan=2)


    def show(self):
        if self.controller.train_x != []:
            self.trainingExamples = self.controller.train_x.shape[0]
            self.trainingParameters = self.controller.train_x.shape[1]
            self.trainingClasses = self.controller.classes

            if self.controller.TrainImagesLoaded == 1:
                if len(self.controller.train_x.shape) == 4: #Check the number of color channels
                    colorChannels = self.controller.train_x.shape[3]
                else:
                    colorChannels = 1
                plotRandomExampleTrain(self, self.controller)
                self.trainingParameters = str(self.controller.train_x.shape[1])+"x"+\
                                          str(self.controller.train_x.shape[2])+"x"+str(colorChannels)
            else:
                self.trainingParameters = self.controller.train_x.shape[1]

        labelTrainNumber = Label(self, text=str(self.trainingExamples), font=("Helvetica", 10))
        labelTrainNumber.grid(row=1, column=0, pady=5, padx=10)

        labelTrainParameters = Label(self, text=str(self.trainingParameters), font=("Helvetica", 10))
        labelTrainParameters.grid(row=1, column=1, pady=5, padx=10)

        labelTrainClasses = Label(self, text=str(self.trainingClasses), font=("Helvetica", 10))
        labelTrainClasses.grid(row=1, column=2, pady=5, padx=10)




"""----------------------------------------------------------------------------------------------------------------------"""


"""----------------------------------------------------------------------------------------------------------------"""
def Return(controller):
    print("Go back to main screen")
    controller.show_frame(StartWindow)

def RefreshTest(self, controller):
    plotRandomExampleTest(self, controller)


def plotRandomExampleTest(self, controller):

    plot_frame = Frame(self, bd=3, relief=SUNKEN)
    plot_frame.grid(column=0, row=2, columnspan=3)

    fig = Figure(figsize=(5, 2.5), dpi=80)
    a = fig.add_subplot(111)
    idx = np.arange(controller.test_x.shape[0])
    choice = np.random.choice(idx, replace=False)
    a.imshow(controller.test_x[choice])

    #DrawingArea
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

class InfoTestWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.testExamples = 0
        self.testParameters = 0
        self.testClasses = 0
        labelNumberInfo = Label(self, text="Number of test examples", font=("Helvetica", 10))
        labelNumberInfo.grid(row=0, column=0, pady=(10, 5), padx=10)
        labelParametersInfo = Label(self, text="Number of test parameters", font=("Helvetica", 10))
        labelParametersInfo.grid(row=0, column=1, pady=(10, 5), padx=10)
        labelClassesInfo = Label(self, text="Number of classes", font=("Helvetica", 10))
        labelClassesInfo.grid(row=0, column=2, pady=(10, 5), padx=10)
        buttonReturn = Button(self, text="Return", command=lambda: Return(self.controller), font=("Agency FB", 14), width=16)
        buttonReturn.grid(row=4, column=0, pady=25, columnspan=2)
        buttonRefresh = Button(self, text="Refresh", command=lambda: RefreshTest(self, self.controller), font=("Agency FB", 14), width=16)
        buttonRefresh.grid(row=4, column=1, pady=25, columnspan=2)

    def show(self):
        if self.controller.test_x != []:
            self.testExamples = self.controller.test_x.shape[0]
            self.testParameters = self.controller.test_x.shape[1]
            self.testClasses = self.controller.classes

            if self.controller.TestImagesLoaded == 1:
                if len(self.controller.test_x.shape) == 4: #Check the number of color channels
                    colorChannels = self.controller.test_x.shape[3]
                else:
                    colorChannels = 1
                plotRandomExampleTest(self, self.controller)
                self.testParameters = str(self.controller.test_x.shape[1])+"x"+\
                                      str(self.controller.test_x.shape[2])+"x"+str(colorChannels)
            else:
                self.testParameters = self.controller.test_x.shape[1]

        labelTestNumber = Label(self, text=str(self.testExamples), font=("Helvetica", 10))
        labelTestNumber.grid(row=1, column=0, pady=5, padx=10)

        labelTestParameters = Label(self, text=str(self.testParameters), font=("Helvetica", 10))
        labelTestParameters.grid(row=1, column=1, pady=5, padx=10)

        labelTestClasses = Label(self, text=str(self.testClasses), font=("Helvetica", 10))
        labelTestClasses.grid(row=1, column=2, pady=5, padx=10)


"""----------------------------------------------------------------------------------------------------------------------"""

class StartWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Configura la ventana inicial
        titulo = tk.Label(self, text="IA Desk", font=("Helvetica", 14, 'bold'))
        titulo.grid(column=0, row=0, pady=10)


"""---------------------------------------------------------------------------------------------------------------------"""


def adamConf(controller):
    print("Go to configuration mode for algorith Adam")
    algorithm = "adam"
    first_dim = controller.get_first_dim()
    controller.frames[WindowConfig].optionsSel(algorithm, first_dim)
    controller.show_frame(WindowConfig)


def gradConf(controller):
    print("Go to configuration mode for algorith Gradient Descent")
    algorithm = "GD"
    first_dim = controller.get_first_dim()
    controller.frames[WindowConfig].optionsSel(algorithm, first_dim)
    controller.show_frame(WindowConfig)

def RMSConf(controller):
    print("Go to configuration mode for algorith RMSProp")
    algorithm = "RMS"
    first_dim = controller.get_first_dim()
    controller.frames[WindowConfig].optionsSel(algorithm, first_dim)
    controller.show_frame(WindowConfig)

def nadamConf(controller):
    print("Go to configuration mode for algorith Nadam")
    algorithm = "nadam"
    first_dim = controller.get_first_dim()
    controller.frames[WindowConfig].optionsSel(algorithm, first_dim)
    controller.show_frame(WindowConfig)


def cancel(controller):
    print("Cancel training go back to main screen")
    controller.enable_menu(0)
    controller.enable_menu(1)
    controller.show_frame(StartWindow)


class AlgorithmSelection(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        titulo = Label(self, text="Select Optimization Algorithm", font=("Helvetica", 14, 'bold'))
        titulo.grid(row=0, column=0, pady=20, columnspan=2)

        buttonGD = Button(self, text="Gradient descent", command=lambda: gradConf(controller), font=("Agency FB", 14), width=16)
        buttonGD.grid(row=2, column=0, pady=15, sticky=W)
        buttonRMS = Button(self, text="RMS Prop", command=lambda: RMSConf(controller), font=("Agency FB", 14), width=16)
        buttonRMS.grid(row=2, column=1, pady=15, sticky=E)
        buttonmomentum = Button(self, text="Nadam", command=lambda: nadamConf(controller), font=("Agency FB", 14), width=16)
        buttonmomentum.grid(row=3, column=0, pady=15, sticky=W)
        buttonadam = Button(self, text="Adam", command=lambda: adamConf(controller), font=("Agency FB", 14), width=16)
        buttonadam.grid(row=3, column=1, pady=15, sticky=E)
        labelInfo = Label(self, text="Info: More algortithm optimizers will be added in later version of the program", font=("Helvetica", 8))
        labelInfo.grid(row=4, column=0, pady=20, columnspan=2)
        buttonCancel = Button(self, text="Cancel", command=lambda: cancelar(controller), font=("Agency FB", 14), width=16)
        buttonCancel.grid(row=5, column=0, pady=25, columnspan=2)


"""--------------------------------------------------------------------------------------------------------------------------"""

def accept(self, controller):


    dict = {'type':self.algorithm, 'dims':self.dimension, 'epochToPrint':int(self.valToPrint.get()), 'epochs':int(self.valEpochs.get()),
            'batch':int(self.valBatchsize.get()),'lambda':float(self.valLambda.get()), 'decay':float(self.valDecay.get()),
            'momentum':float(self.valMomentum.get()), 'RMSmomentum':float(self.valRMSMomentum.get()), 'beta1':float(self.valBeta1.get()),
            'beta2':float(self.valBeta2.get()), 'AdamEpsilon':float(self.valAdamEpsilon.get()), 'RMSEpsilon':float(self.valRMSEpsilon.get()),
            'x': controller.train_x, 'y': controller.train_y, 'x_test': controller.test_x, 'y_test': controller.test_y, 'classes': controller.classes}

    if (self.selDim.get() == 1):
            print("Going to network dimensions creator")
            controller.show_frame(DimCreator)
            controller.frames[DimCreator].passDim(dict)
    else:

        if len(self.dimension) > 1:
            controller.show_frame(TrainWindow)
            controller.frames[TrainWindow].launch(dict)


        else:

            messagebox.showerror("Error", "net dimension hasn't been loaded")

def cancelar(controller):
    print("Cancel training go back to main screen")
    controller.enable_menu(0)
    controller.enable_menu(1)
    controller.show_frame(StartWindow)


def load(self):
    print("Loading predefined network")

    path = filedialog.askopenfilename(initialdir="./", title="Save NN",
                                               filetypes=(("Text File", "*.txt"), ("all files", "*.")))
    if path != None:
        new_path, extension = os.path.splitext(path)
        print("File: --" + new_path + "-- loaded sucesfully")
        print("extension " + extension)
        file = open(path, 'r')
        for linea in file.readlines():
            self.dimension.append(int(linea))
        self.valFiledim.set(path)
        file.close()


class WindowConfig(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create the variables that configure the net
        self.valFiledim = StringVar(value='')
        self.valToPrint = StringVar(value='100')
        self.selDim = IntVar()
        self.valBatchsize = StringVar(value='32')
        self.valEpochs = StringVar(value='1500')
        self.valLambda = StringVar(value='0.0001')
        self.valDecay = StringVar(value='0.9')
        self.valMomentum = StringVar(value='0.9')
        self.valRMSMomentum = StringVar(value='0.0')
        self.valBeta1 = StringVar(value='0.9')
        self.valBeta2 = StringVar(value='0.999')
        self.valRMSEpsilon = StringVar(value='1e-10')
        self.valAdamEpsilon = StringVar(value='1e-8')
        self.scheduledDecay = StringVar(value='0.004')

        labelConfig = Label(self, text="Configuration", font=("Helvetica", 14, 'bold'))
        labelConfig.grid(row=0, column=0, pady= 10, columnspan=3)

        # Indicate if network dimensions will be loaded or if new one will dbe created
        self.selDim.set(1)
        labelReg = Label(self, text="Load or create dimension of NN", font=("Helvetica", 10))
        labelReg.grid(row=1, column=0, columnspan=2, pady=5, sticky=W)
        labelFile = Label(self, textvariable=self.valFiledim, font=("Helvetica", 8))
        labelFile.grid(row=4, column=0, columnspan=3)
        rdbCrear = Radiobutton(self, text="Create", value=1, variable=self.selDim)
        rdbCrear.grid(row=2, column=0, sticky=W)
        rdbCargar = Radiobutton(self, text="Load", value=2, variable=self.selDim)
        rdbCargar.grid(row=3, column=0, sticky=W)

        # Options to load the network dimensions
        buttonLoad = Button(self, text="Load dimensions", command=lambda: load(self), font=("Helvetica", 10))
        buttonLoad.grid(row=2, column=1, rowspan=2, sticky=W)

        # Define each how many epochs loss and accuracy values will be printed
        labelToPrintEpochs = Label(self, text="Print metrics after Nª epoch:", font=("Helvetica", 10))
        labelToPrintEpochs.grid(row=5, column=0, sticky=W, pady= (0,2))
        toPrintEpochs = Entry(self, textvariable=self.valToPrint, width=10)
        toPrintEpochs.grid(row=6, column=0, sticky=W)

        # Define the size of mini-batch
        labelBatchSize = Label(self, text="Mini-batch size", font=("Helvetica", 10), width=10)
        labelBatchSize.grid(row=5, column=2, sticky=W, pady= (0,2))
        batchSize = Entry(self, textvariable=self.valBatchsize, width=10)
        batchSize.grid(row=6, column=2, sticky=W)

        # Define the number of epochs
        labelEpochs = Label(self, text="Nº Epochs", font=("Helvetica", 10))
        labelEpochs.grid(row=7, column=0, sticky=W, pady= (5,2))
        epochs = Entry(self, textvariable=self.valEpochs, width=10)
        epochs.grid(row=8, column=0, sticky=W)

        # Define Lambda or learnig rate
        labelLamba = Label(self, text="Learning Rate", font=("Helvetica", 10))
        labelLamba.grid(row=7, column=2, sticky=W, pady=(5, 2))
        landa = Entry(self, textvariable=self.valLambda, width=10)
        landa.grid(row=8, column=2, sticky=W)

        buttonAccept = Button(self, text="Accept", command=lambda: accept(self,controller), font=("Helvetica", 12),width=8)
        buttonAccept.grid(row=13, column=0, sticky=SE, pady= (5,2))
        buttonCancel = Button(self, text="Cancel", command=lambda: cancelar(controller), font=("Helvetica", 12), width=8)
        buttonCancel.grid(row=13, column=2, sticky=SW, pady= (5,2))

    def optionsSel(self, algorithm, first_dim):

        self.dimension = [first_dim]
        self.algorithm = algorithm

        if algorithm == "RMS":

            #Define epsilon for RMSProp algorithm
            labelEpsilon = Label(self, text="Epsilon(Def recommended)", font=("Helvetica", 10))
            labelEpsilon.grid(row=9, column=0, sticky=W, pady=(5, 2))
            epsilonRMS = Entry(self, textvariable=self.valRMSEpsilon, width=10)
            epsilonRMS.grid(row=10, column=0, sticky=W)

            # Define decay for the RMSProp algorithm
            labelDecayRMS = Label(self, text="Decay(Def recommended)", font=("Helvetica", 10))
            labelDecayRMS.grid(row=9, column=2, sticky=W, pady=(5, 2))
            decayRMS = Entry(self, textvariable=self.valDecay, width=10)
            decayRMS.grid(row=10, column=2, sticky=W)

        elif algorithm == "nadam":

            # Define beta1 for NADAM algorithm
            labelBeta1Nadam = Label(self, text="Beta1(Def recommended)", font=("Helvetica", 10))
            labelBeta1Nadam.grid(row=9, column=0, sticky=W, pady=(5, 2))
            beta1Nadam = Entry(self, textvariable=self.valBeta1, width=10)
            beta1Nadam.grid(row=10, column=0, sticky=W)

            # Define beta2 for NADAM algorithm
            labelBeta2Nadam = Label(self, text="Beta2(Def recommended)", font=("Helvetica", 10))
            labelBeta2Nadam.grid(row=9, column=2, sticky=W, pady=(5, 2))
            beta2Nadam = Entry(self, textvariable=self.valBeta2, width=10)
            beta2Nadam.grid(row=10, column=2, sticky=W)

            # Define Epsilon for NADAM algorithm
            labelEpsilonNadam = Label(self, text="Epsilon(Def recommended)", font=("Helvetica", 10))
            labelEpsilonNadam.grid(row=11, column=0, sticky=W, pady=(5, 2))
            epsilonNadam = Entry(self, textvariable=self.valAdamEpsilon, width=10)
            epsilonNadam.grid(row=12, column=0, sticky=W)

            # Define Scheduled Decay for NADAM algorithm
            labelScheduledDecay = Label(self, text="ScheduledDecay(Def recommended)", font=("Helvetica", 10))
            labelScheduledDecay.grid(row=11, column=2, sticky=W, pady=(5, 2))
            Scheduleddecay = Entry(self, textvariable=self.scheduledDecay, width=10)
            Scheduleddecay.grid(row=12, column=2, sticky=W)

        elif algorithm == "adam":

            # Define beta1 for ADAM algorithm
            labelBeta1Adam = Label(self, text="Beta1(Def recommended)", font=("Helvetica", 10))
            labelBeta1Adam.grid(row=9, column=0, sticky=W, pady=(5, 2))
            beta1Adam = Entry(self, textvariable=self.valBeta1, width=10)
            beta1Adam.grid(row=10, column=0, sticky=W)

            # Define beta2 for ADAM algorithm
            labelBeta2Adam = Label(self, text="Beta2(Def recommended)", font=("Helvetica", 10))
            labelBeta2Adam.grid(row=9, column=2, sticky=W, pady=(5, 2))
            beta2Adam = Entry(self, textvariable=self.valBeta2, width=10)
            beta2Adam.grid(row=10, column=2, sticky=W)

            # Define Epsilon for ADAM algorithm
            labelEpsilonAdam = Label(self, text="Epsilon(Def recommended)", font=("Helvetica", 10))
            labelEpsilonAdam.grid(row=11, column=0, sticky=W, pady=(5, 2))
            epsilonAdam = Entry(self, textvariable=self.valAdamEpsilon, width=10)
            epsilonAdam.grid(row=12, column=0, sticky=W)




"""-------------------------------------------------------------------------------------------------------"""

def print_options(self, train_accuracy, test_accuracy, model):
    labeltrainaccuraccy = Label(self, text="Train Accuracy:"+str(round(train_accuracy, 2)), font=("Helvetica", 10))
    labeltrainaccuraccy.grid(column=0, row=1, pady=(15,0))
    labeltestaccuraccy = Label(self, text="Test Accuracy:"+str(round(test_accuracy, 2)), font=("Helvetica", 10))
    labeltestaccuraccy.grid(column=1, row=1, pady=(15,0))
    saveParameters = Button(self, text="Save Parameters", command=lambda: save_Parameters_keras(model), font=("Helvetica", 10))
    saveParameters.grid(column=0, row=2, pady=(15,0))
    saveModel = Button(self, text="Save Model", command=lambda: save_Model_keras(model), font=("Helvetica", 10))
    saveModel.grid(column=1, row=2, pady=(15, 0))

class TrainWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controler = controller

    def launch(self, dict):
        print("Launching training")

        # Initialize window texts
        self.valCost = StringVar()
        self.valAcc = StringVar()
        self.valCost.set("Cost after epoch -: -")
        self.valAcc.set("Accuracy after epoch -: -")
        labelTrain = Label(self, text="Training the neural network", font=("Helvetica", 12))
        labelTrain.grid(row=0, column=0, pady=(100,2), columnspan=2)
        labelCosts = Label(self, textvariable=self.valCost, font=("Helvetica", 12))
        labelCosts.grid(row=1, column=0, columnspan=2)
        labelAcc = Label(self, textvariable=self.valAcc, font=("Helvetica", 12))
        labelAcc.grid(row=2, column=0, columnspan=2)

        classes = dict['classes']
        X_train = dict['x']
        Y_train = dict['y']
        X_test = dict['x_test']
        Y_test = dict['y_test']

        #Y_train = convert_to_one_hot(Y_train_orig, classes) 
        #Y_test = convert_to_one_hot(Y_test_orig, classes)

        history, test_cost, test_accuracy, model = model_train(self, X_train, Y_train, X_test, Y_test, classes, dict)

        history_costs = history.history['loss']
        history_accuracy = history.history['acc']
        train_accuracy = history_accuracy[len(history_accuracy)-1]

        # plot the cost
        labelCosts.destroy()
        labelTrain.destroy()
        labelAcc.destroy()
        self.controler.enable_menu(0)
        self.controler.enable_menu(1)
        plot_graph_keras(self, history_costs, history_accuracy)
        print_options(self, train_accuracy, test_accuracy, model)


""""-------------------------------------------------------------------------------------------------------------------"""
def loadModel(self):
    print("LoadingModel")
    path = filedialog.askopenfilename(initialdir="./", title="Select file",
                                      filetypes=(("HDF5 files", "*.h5"), ("pikle files", "*.pkl")))

    if path != None:
        new_path, extension = os.path.splitext(path)
        print("File: --" + new_path + "-- loaded succesfully")
        print("extension " + extension)
        self.model = load_model(path)
        self.valFileIn.set(path)
    else:
        print("File load error")

def loadInput(self):
    print("Loading example")
    path = filedialog.askopenfilename(initialdir="./", title="Select file",
                                      filetypes=(("numpy files", "*.npy"),("dat files", "*.dat"),
                                                 ("csv files", "*.csv")))

    if path != None:
        new_path, extension = os.path.splitext(path)
        print("File: --" + new_path + "-- loaded succesfully")
        print("extension " + extension)
        self.Input = np.load(path)
        print(self.Input)
        self.valFileIn.set(path)

    else:
        print("File load error")

def MakePrediction(self):
    print("Calculating prediction")
    #TODO


class PredictionWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controler = controller
        self.valFileMod = StringVar(value='None')
        self.valFileIn = StringVar(value='None')
        self.model = None
        self.Input = ""

        labelMakePred = Label(self, text="Make a prediction", font=("Helvetica", 14, 'bold'))
        labelMakePred.grid(row=0, column=0, columnspan=3, pady=(30,30))
        # Load a model
        labelModel = Label(self, text="Load NN model", font=("Helvetica", 10))
        labelModel.grid(row=1, column=0, padx=(0,20))
        buttonLoadModel = Button(self, text="Load", command=lambda: loadModel(self), width= 10, font=("Helvetica", 10))
        buttonLoadModel.grid(row=2, column=0,  padx=(0,20))

        labelInput = Label(self, text="Load Input", font=("Helvetica", 10))
        labelInput.grid(row=1, column=2,  padx=(20,0))
        buttonLoadInput = Button(self, text="Load", command=lambda: loadModel(self), width=10, font=("Helvetica", 10))
        buttonLoadInput.grid(row=2, column=2,  padx=(20,0))

        labelFileMod = Label(self, textvariable=self.valFileMod, font=("Helvetica", 8))
        labelFileMod.grid(row=3, column=0, pady=10, columnspan=3)
        labelFileIn = Label(self, textvariable=self.valFileIn, font=("Helvetica", 8))
        labelFileIn.grid(row=4, column=0, pady=10, columnspan=3)

        buttonPredict = Button(self, text="Predict", command=lambda: MakePrediction(self), width=10, font=("Helvetica", 10))
        buttonPredict.grid(row=5, column=1)

""""-------------------------------------------------------------------------------------------------------------------"""

def divideSet(self, controller):
    print("Split dataset")
    rawData = controller.rawData
    test_size = float(self.valPerc.get())/100
    #y = rawData[:, (rawData.shape[1]-1)]
    y = rawData[:, -1]
    y = y.astype('int32')
    x = rawData[:, 0:rawData.shape[1]-1]
    controller.classes = len(np.unique(y))
    controller.train_x, controller.test_x, controller.train_y, controller.test_y = train_test_split(x, y,
                                                                test_size=test_size, random_state=2018)

    print("Train dataset dimensions: ",controller.train_x.shape)
    print("Test dataset dimensions: ", controller.test_x.shape)
    print("Classes: ", controller.classes)
    controller.enable_menu(0)
    controller.enable_menu(1)
    controller.show_frame(StartWindow)

class SetCreation(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.valPerc = StringVar(value='15')

        selectorlabel = Label(self, text="Select the test set percentage", font=("Helvetica", 12))
        selectorlabel.grid(row=0, column=0, pady=(100, 10), columnspan=3)
        setPercentage = Entry(self, textvariable=self.valPerc, width=10)
        setPercentage.grid(row=1, column=1)
        buttonDivide = Button(self, text="Split", command=lambda: divideSet(self, controller), font=("Agency FB", 14), width=10)
        buttonDivide.grid(row=2, column=1, pady=15)

