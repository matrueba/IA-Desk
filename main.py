import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from Windows import *
from NNhandler import *
from TFfunctions import *
from HelpWindows import *
from AuxWindows import *
from LoadFunctions import *
import NNhandler as handler

def center_screen(main):
    w = main.winfo_screenwidth()
    h = main.winfo_screenheight()
    x = int(w / 2 - 300)
    y = int(h / 2 - 200)
    return x, y

def InfoTrainSet(self):
    self.show_frame(InfoTrainWindow)
    self.frames[InfoTrainWindow].show()

def InfoTestSet(self):
    self.show_frame(InfoTestWindow)
    self.frames[InfoTestWindow].show()

def Train(self):
    # if self.train_x and self.test_x:
    print("Go to algorithm selection window")
    self.show_frame(AlgorithmSelection)
    self.disable_menu(0)
    self.disable_menu(1)
    # else:
    # messagebox.showerror("Error", "no set has been loaded")

def TabsFrame(self):
    self.show_frame(Tabs)
    Tabs.option_add(self)
    self.disable_menu(0)
    self.disable_menu(1)

def about():
    ab = About()
    ab.mainloop()

def Tutorial():
    tutorial = TutorialWindow()
    tutorial.mainloop()

def split_raw_set(self):

    print("Go to raw set creation section")
    
    self.rawData = load_raw(self)
    self.show_frame(SetCreation)
    self.disable_menu(0)
    self.disable_menu(1)
    """if (self.rawData != None):
        self.show_frame(SetCreation)
        self.disable_menu(0)
        self.disable_menu(1)
    else:
        messagebox.showerror("Error no file loaded")
    """

def makePrediction(self):
    print("Make prediction")

    self.show_frame(PredictionWindow)
    self.disable_menu(0)
    self.disable_menu(1)


class Main(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Neural Desk")

        container = tk.Frame(self)
        container.pack()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Initialize Empty Sets
        self.TrainImagesLoaded = 0
        self.TestImagesLoaded = 0
        self.classes = 0
        self.rawData = np.array([])
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])

        # Menu bar creation
        self.menubar = tk.Menu(self, tearoff=0)
        # Menu creation
        menuGeneral = tk.Menu(self.menubar, tearoff=0)
        menuLoad = tk.Menu(self.menubar, tearoff=0)
        menuTest = tk.Menu(self.menubar, tearoff=0)
        menuAbout = tk.Menu(self.menubar, tearoff=0)
        # Menu Commands
        menuGeneral.add_command(label='Train', command=lambda: Train(self))
        menuGeneral.add_separator()
        menuGeneral.add_command(label='Info Train Set', command=lambda: InfoTrainSet(self))
        menuGeneral.add_separator()
        menuGeneral.add_command(label='Info Test Set', command=lambda: InfoTestSet(self))
        menuLoad.add_command(label='Load Raw Dataset', command=lambda: split_raw_set(self))
        menuLoad.add_separator()
        menuLoad.add_command(label='Load Train Set', command=lambda: load_train(self))
        menuLoad.add_separator()
        menuLoad.add_command(label='Load Test Set', command=lambda: load_test(self))
        menuLoad.add_separator()
        menuLoad.add_command(label='Load Images Train Set', command=lambda:load_images_train(self))
        menuLoad.add_separator()
        menuLoad.add_command(label='Load Images Test Set', command=lambda: load_images_test(self))
        menuLoad.add_separator()
        menuLoad.add_command(label='Load Internal Datasets', command=lambda: load_internal_datasets(self))
        menuLoad.add_separator()
        menuLoad.add_command(label='Load Pretrained', command=lambda: loadPretrained(self))
        # menuCargar.add_separator()
        # menuCargar.add_command(label='Tabs', command=lambda: TabsFrame(self))
        menuTest.add_command(label='Make a predition', command=lambda: makePrediction(self))
        menuAbout.add_command(label='Help', command=Tutorial)
        menuAbout.add_separator()
        menuAbout.add_command(label='About', command=about)
        # Add menus to menu bar
        self.menubar.add_cascade(label='General', menu=menuGeneral)
        self.menubar.add_cascade(label='Load files', menu=menuLoad)
        self.menubar.add_cascade(label='Test', menu=menuTest)
        self.menubar.add_cascade(label='About', menu=menuAbout)
        
        tk.Tk.config(self, menu=self.menubar)

        self.frames = {}

        for F in (
        StartWindow, AlgorithmSelection, WindowConfig, DimCreator, TrainWindow, PredictionWindow, SetCreation, Tabs, InfoTrainWindow, InfoTestWindow):
            frame = F(container, self)

            self.frames[F] = frame

        self.show_frame(StartWindow)

    def show_frame(self, cont):
        # This piece of code is included to forget the grid of previous frames. Bacuse if not the frame
        # starts with the configuration of grid created the last time
        # critical above all for matplotlib
        for f in self.frames.values():  
            f.grid_forget() 

        frame = self.frames[cont]
        frame.grid(row=0, column=0,
                   sticky='nsew')  # sticky nsew para que el nuevo frame se expanda coloque encima del otro
        frame.tkraise()

    def disable_menu(self, index):
        self.menubar.entryconfig(index, state="disabled")

    def enable_menu(self, index):
        self.menubar.entryconfig(index, state="normal")

    def get_first_dim(self):
        return self.train_x.shape[0]
