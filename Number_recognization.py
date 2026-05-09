import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#this tells the Python just where the file is located as im having a issue importing class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NetBrain import NeuralBrain

x1 = np.array([[1,0,1,1],[0,1,0,1]]) #2+3 and 1+1 on binary
y1 = np.array([[1,0,1],[0,1,0]]) # five in binary

model = NeuralBrain(x1,y1,input_size=4 , hidden_size=10 , output_size=3,lr=0.01)
model.train(1000000)
final=model.forward_propagation([[1,0,1,1]])
print(final)
model.graphplot()