import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gzip #to handle and decompress data
import cv2 #for opatmized image processing
import time
import threading # allows alloation of threads and lets code work parallely
from NetBrain import NeuralBrain

#this tells the Python just where the file is located as im having a issue importing class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dir_path = os.path.dirname(os.path.abspath(__file__)) # get absolute path of the folder
file_path = os.path.basename(__file__)# file path
file_name = os.path.join(dir_path,f"{os.path.splitext(file_path)[0]}.npz")# remove extension

def loading_screen(duration=3):
    print("Initializing NeuralBrain...")
    chars = ["/", "-", "\\", "|"] # The rotating animation
    end_time = time.time() + duration
    
    while time.time() < end_time:
        for char in chars:
            # \r moves the cursor to the start of the line
            sys.stdout.write(f"\rLoading {char} ")
            sys.stdout.flush() # to load the line immediately 
            time.sleep(0.1)

def parallel_screen(stop_command):
    chars = ["/", "-", "\\", "|"] # The rotating animation
    while not stop_command.is_set():
        for c in chars:
            sys.stdout.write(f"\rTraining NN {c}")
            sys.stdout.flush()
            time.sleep(0.1)#delay
    sys.stdout.write("Training Complete")

# Assigning a thread event
stop_spinning = threading.Event()



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
while(True):
    os.system('cls')
    loading_screen()
    print("Here are the options:\n1) Train file\n2) Run Trained file\n3) Remove Trained Data\n")

    ask = int(input("Your Answer: "))
    match ask:
        case 1:
            stop_spinning.clear() # clears set if True
            # Define what threading action is to be assigned
            t = threading.Thread(target=parallel_screen,args=(stop_spinning,)) # function and set args
            t.start() # start spinning animation
            try:
                model.train(1000000)
            finally:
                stop_spinning.set() # Returns True for while loop to stop
                t.join() # Ensures the seperated threads join to acts on the main 
            
            model.learn()
            model.graphplot()

        
        case 2:
            model.remember()
            a = int(input("Enter a number "))
            final=model.forward_propagation([[a]])
            print(final)
            print("Softmax: ")
            print(model.softmax)

        case 3:
            ask = int(input("Do you really want to delete(1/0)"))
            if ask == 1:
                loading_screen()
                model.forget()
        
        case _:
            print("Wrong input:-->")
            continue
    
    ask = int(input("101 to exit -->"))
    if ask == 101:
        break

