# Import and run the GUI
from src.gui.gui import MazeTrainingGUI
import tkinter as tk

def main():
    root = tk.Tk()
    app = MazeTrainingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()