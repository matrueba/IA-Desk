import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from NNhandler import *
import Windows as nw
import os


# Insert a layer with its dimensions to network
def insert_layer(self):
    self.list.insert(END, "Layer " + str(self.index) + " dimension: " + self.dimension.get())
    self.arraydim.append(self.dimension.get())
    self.index = self.index + 1


# Confirm the network layers
def confirm(self, controller):
    print("Dimension layers: " + str(self.arraydim))
    print("Number of layers: " + str(len(self.arraydim) - 1))
    self.dict['dims'] = self.arraydim
    controller.show_frame(nw.TrainWindow)
    controller.frames[nw.TrainWindow].launch(self.dict)


# Save the dimensions of the network in file
def save_layer(self):
    path = filedialog.asksaveasfilename(initialdir="./", title="Save NN layers, don't forget put the right extension",
                                        filetypes=(("Text File", "*.txt"), ("all files", "*.")))

    if path != None:

        print("Path to save " + path)

        file = open(path, 'w')
        for i in range(len(self.arraydim)):
            print(str(self.arraydim[i]))
            file.seek(0, 2)
            file.write(self.arraydim[i] + "\n")
        file.close()

        print("Neural network dimensions saved succesfully")

    else:

         print("File save error")


# Delete a layer of network
def delete(self):
    self.list.delete(END)
    self.arraydim.pop()
    self.index = self.index - 1

def cancel(self, controller):
    print("Cancel training and go back")
    controller.show_frame(nw.WindowConfig)


class DimCreator(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create the container that contains list and scroll
        container = Canvas(self)
        container.grid(column=0, row=2, columnspan=3, pady=(20,10))

        # Create the variables to use 
        self.arraydim = []
        self.entry = IntVar() 
        self.index = 1  

        # Crea the set field for the dimensions and label
        label = Label(self, text="Layer dimension")
        label.grid(column=0, row=0, sticky=W, pady=(20,2))
        self.dimension = Entry(self, width=7)
        self.dimension.grid(column=0, row=1, sticky=W, pady=(2,2))

        # Create the screen buttons
        insert = Button(self, text="Insert", command=lambda: insert_layer(self), width=8)
        insert.grid(column=1, row=1)
        remove = Button(self, text="Delete", command=lambda: delete(self), width=8)
        remove.grid(column=2, row=1)
        accept = Button(self, text="Accept", command=lambda: confirm(self, controller), width=10)
        accept.grid(column=0, row=3, pady=(15,0), sticky=W)
        save = Button(self, text="Save Net", command=lambda: save_layer(self), width=10)
        save.grid(column=1, row=3, pady=(15,0), padx=(0,10), sticky=W)
        back = Button(self, text="Cancel", command=lambda: cancel(self, controller), width=10)
        back.grid(column=2, row=3, pady=(15,0))

        # Create the scroll and set it into container
        scrollbar = Scrollbar(container, orient=VERTICAL)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Create the listbox and set it into container
        self.list = Listbox(container, width=40, height=14, borderwidth=2, yscrollcommand=scrollbar.set)
        self.list.pack()

        scrollbar.config(command=self.list.yview)


    def passDim(self, dict):

        self.arraydim = dict['dims'] 
        self.dict = dict

        

