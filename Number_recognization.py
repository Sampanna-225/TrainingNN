import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gzip #to handle and decompress data
import cv2 #for opatmized image processing

#this tells the Python just where the file is located as im having a issue importing class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dir_path = os.path.dirname(os.path.abspath(__file__)) # get absolute path of the folder
file_path = os.path.basename(__file__)# file path
file_name = os.path.join(dir_path,f"{os.path.splitext(file_path)[0]}.npz")# remove extension

from NetBrain import NeuralBrain

x1 = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
]) 
y1 = np.array([[1,0,0,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0,0,0],
               [0,0,0,0,1,0,0,0,0,0],
               [0,0,0,0,0,1,0,0,0,0],
               [0,0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,0,0,1]]) # from 1 to 10

sets = [1,2,3,4,5,6,7,8,10]

model = NeuralBrain(x1,y1,input_size=1 , hidden_size=10 , output_size=10,lr=0.01)

model.remember()
model.train(1000000)

a = int(input("Enter a number "))

final=model.forward_propagation([[a]])
model.learn()
print(final)
print(model.softmax(final))
print(f"The guessed number is {sets[np.argmax(model.softmax(final))]}")
model.graphplot()  