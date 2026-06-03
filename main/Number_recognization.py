import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gzip #to handle and decompress data
import cv2 #for opatmized image processing
import time
import threading # allows alloation of threads and lets code work parallely
from tqdm import tqdm

#this tells the Python just where the file is located as im having a issue importing class adds location of parent file
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dir_path = os.path.dirname(os.path.abspath(__file__)) # get absolute path of the folder
parent_dir = os.path.dirname(dir_path)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)#To avoid exploitation

file_path = os.path.basename(__file__)# file path
file_name = os.path.join(dir_path,f"{os.path.splitext(file_path)[0]}.npz")# remove extension

#image
data_folder = os.path.join(parent_dir,"data")


from brain.NetBrain import NeuralBrain
from ui.animation import loading_screen
from ui.animation import parallel_screen
from ui.animation import dot_animation
from data.unzip import X_image
from data.unzip import X_image_test
from data.unzip import Y_label
from data.unzip import y_label_test
from main.image import image_converter
# Assigning a thread event
stop_spinning = threading.Event()



# x1 = np.array([
#     [1],
#     [2],
#     [3],
#     [4],
#     [5],
#     [6],
#     [7],
#     [8],
#     [9],
#     [10]
# ]) 
# y1 = np.array([[1,0,0,0,0,0,0,0,0,0],
#                [0,1,0,0,0,0,0,0,0,0],
#                [0,0,1,0,0,0,0,0,0,0],
#                [0,0,0,1,0,0,0,0,0,0],
#                [0,0,0,0,1,0,0,0,0,0],
#                [0,0,0,0,0,1,0,0,0,0],
#                [0,0,0,0,0,0,1,0,0,0],
#                [0,0,0,0,0,0,0,1,0,0],
#                [0,0,0,0,0,0,0,0,1,0],
#                [0,0,0,0,0,0,0,0,0,1]]) # from 1 to 10
x1 =X_image
y1 =Y_label
sets = [1,2,3,4,5,6,7,8,9] # for normal number guessing
sets_adv = [0,1,2,3,4,5,6,7,8,9] # for MINST label

model = NeuralBrain(x1,y1,input_size=784 , hidden_size=128 , output_size=10,lr=0.1)
while(True):
    os.system('cls')
    loading_screen()
    print("Here are the options:\n1) Train file\n2) Run Trained file\n3) Remove Trained Data\n4) To Exit")
    try:
        ask = int(input("Your Answer: "))
    except:
        print("Error Enter Number!!!")
    match ask:
        case 1:
            lo = int(input("Enter training epoch: "))
            # stop_spinning.clear() # clears set if True
            # # Define what threading action is to be assigned
            # t = threading.Thread(target=parallel_screen,args=(stop_spinning,)) # function and set args
            # t.start() # start spinning animation
            # try:
            model.train(lo)
            # finally:
            #     stop_spinning.set() # Returns True for while loop to stop
            #     t.join() # Ensures the seperated threads join to acts on the main 
            
            model.learn()
            model.graphplot()
            continue

        
        case 2:
            model.remember()
            try:
               a =input("Enter a image number: ")
               test_img = os.path.join(data_folder,f"{a}.png")
               temp =  image_converter(test_img)
               
               answer = sets_adv[np.argmax(model.forward_propagation(temp))]
               print(f"\n The Prediction is {answer}\n")
               print(model.forward_propagation(temp))
               print(test_img)
               
            except FileNotFoundError as e:
                print(e)
                
        case 3:
            
            try:
                 ask = int(input("Do you really want to delete(1/0) : "))
                 if ask == 1:
                    loading_screen()
                    model.forget()
                    continue
                 
            except:
                 print("Enter a number !!!!!")

            
        
        case 4:
            dot_animation()
            break

        case _:
            print("Wrong input:-->")
            print("Restarting")
            time.sleep(1)
            continue
    
    try:
        ask = input("101 to exit --> or press any to continue")
        if ask == "101":
            break

    except:
        continue

