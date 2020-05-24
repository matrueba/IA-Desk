import tkinter as tk

class TutorialWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        tk.Tk.wm_title(self, "Tutorial")
        container = tk.Frame(self)
        container.pack()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        label = tk.Label(container, text="This module is under development")
        label.grid(row=0, column=0, pady=(20, 5))


class About(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        container = tk.Frame(self)
        container.pack()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        labelVer = tk.Label(container, text="Ver 1.0")
        labelVer.grid(row=1, column=0, pady=(0, 15))
