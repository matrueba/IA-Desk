from main import *

def launcher():
    main = Main()
    x, y = center_screen(main)
    main.geometry("600x400" + "+" + str(x) + "+" + str(y))
    main.mainloop()


if __name__ == "__main__":
    launcher()
