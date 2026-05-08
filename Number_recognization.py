import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#this tells the Python just where the file is located as im having a issue importing class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NetBrain import NeuralBrain

x1 = np.array([[0,0],[0,0],[1,1],[0,1]]) #2+3 on binary
y1 = np.array([[0],[1],[0],[1]]) # five in binary

model = NeuralBrain(x1,y1,input_size=2 , hidden_size=4 , output_size=3,lr=0.1)
model.train(10000)
_,final=model.forward_propagation(x1)
model.graphplot()