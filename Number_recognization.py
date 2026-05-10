import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#this tells the Python just where the file is located as im having a issue importing class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

model = NeuralBrain(x1,y1,input_size=1 , hidden_size=10 , output_size=10,lr=0.01)
model.train(1000000)
a = int(input("Enter a number"))
final=model.forward_propagation([[a]])
print(final)
model.graphplot()