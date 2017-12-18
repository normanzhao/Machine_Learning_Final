import numpy as np
import skimage
import skimage.io
import keras
from keras.models import load_model
import keras.backend as K
#clear session to ensure proper runtime
K.clear_session()

#change all images to 30x30
from skimage.transform import resize
def resizer(img):
    return resize(img,(30,30),mode='constant')

#load model created ftom model.ipynb
model = load_model('numeric_data.h5')

#import Tkinter to create a UI
from tkinter import *

#matrix of 300x300 to be rescaled to 30x30
pred_img =  np.zeros((300,300))
master = Tk()

#LABEL to display messages, CLEAR to set flag to wipe canvases and reset matrix
LABEL = StringVar()
CLEAR = BooleanVar()

master.minsize(width=600, height=325)
master.resizable(width=False, height=False)
   
#input canvas
canvas = Canvas(master, width=300, height=300, bg="white")
canvas.grid(row=0, column=0, rowspan=20)

#output canvas
output = Canvas(master, width=300, height=300)
output.grid(row=0, column=1, rowspan=20)

Label(master, textvariable=LABEL).grid(row=20, column=0, columnspan=4)

#draw by holding down m1 and moving mouse, set boundaries so matrix indices don't get out of range.
#works by flipping 0s(white) to 1s(black) on a 10x10 square around the current mouse cursor position
def draw(event):
    LABEL.set("")
    canvas.create_rectangle(event.x-5, event.y-5, event.x+5, event.y+5, fill="black")
    for xval in range(int(event.x-5), int(event.x+6)):
        for yval in range(int(event.y-5), int(event.y+6)):
            if 0 < xval and 300> xval and 0 < yval and 300> yval:
                pred_img[yval,xval] = 1
    pred = model.predict_classes(resizer(pred_img).reshape((900))[None,:])[0]
    output.delete("all")
    output.create_text(150,150,text=pred, font="Times 150 italic bold")

#clear input and output UI elements set flag to clear matrix
def clear(event):
    canvas.delete("all")
    output.delete("all")
    LABEL.set("Cleared input")
    CLEAR.set(True)

canvas.bind("<Button-1>", draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<Button-3>", clear)

#infinite loop with flag check to clear matrix
while True:
    if(CLEAR.get()):
        pred_img =  np.zeros((300,300))
        CLEAR.set(False)
    master.update()