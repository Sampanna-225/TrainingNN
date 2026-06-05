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
from data.unzip import X_titanic
from data.unzip import Y_titanic
from data.unzip import a
from data.unzip import f
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

sets = [1,2,3,4,5,6,7,8,9] # for normal number guessing
sets_adv = [0,1,2,3,4,5,6,7,8,9] # for MINST label
titanic_sets =["They Will Not Survive.", "They Will Survive"]

x1 =X_image
y1 =Y_label

x2 =X_titanic
y2 =Y_titanic

model_image = NeuralBrain(x1,y1,input_size=784 , hidden_size=128 , output_size=10,lr=0.1)
model_prediction = NeuralBrain(x2,y2,input_size=6 , hidden_size=4 , output_size=2,lr=0.1)
loading_screen()
while(True):
    
    
    print("Here are the options:\n1) Train Image (BETA)\n2 Train Titanic Survival Prediction \n3) Run Trained file\n4) Remove Trained Data\n5) To Exit")
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
            model_image.train(lo)
            # finally:
            #     stop_spinning.set() # Returns True for while loop to stop
            #     t.join() # Ensures the seperated threads join to acts on the main 
            
            model_image.learn()
            model_image.graphplot()
            continue

        case 2:
            lo =int(input("Enter training epoch: "))
            model_prediction.train(lo)
            model_prediction.learn()
            model_prediction.graphplot()


        case 3:
            os.system('cls')
            print("1) Pass Image\n2)Pass Titanic Prediction")
            ask = int(input("Whats you choice: "))
            model_image.remember()
            try:
                if ask == 1:
                    a =input("Enter a image number: ")
                    test_img = os.path.join(data_folder,f"{a}.png")
                    temp =  image_converter(test_img)

                    answer = sets_adv[np.argmax(model_image.forward_propagation(temp))]
                    print(f"\n The Prediction is {answer}\n")
                    print(model_image.forward_propagation(temp))
                    print(test_img)
                elif ask == 2:
                    pclass = int(input("Enter 1 "))
                    pclass = pclass if (pclass <= 3 and pclass >=1) else 1
                    sex = int(input("Enter Sex (1-Female/0-Male): "))
                    sex = sex if (sex>=0 and sex<=1 ) else 0
                    age = int(input("Enter age : "))
                    age = age if age > 0 else (-age+1)
                    fare = float(input("Enter fare: "))
                    fare = fare if fare>0 else (-fare +1)
                    pclass_encoding = [0.0,0.0,0.0]
                    pclass_encoding[int(pclass)-1] = 1.0

                    age = (age-min(a))/(max(a)-min(a))
                    age = age if age >= 0 else 0.0 #if the minmax really cause trouble
                    fare = (fare-min(f))/(max(f)-min(f))
                    fare = fare if fare >=0 else 0.0 #just incase negative
                    take = [[pclass_encoding[0],pclass_encoding[1],pclass_encoding[2],sex,age,fare]]
                    takem = np.array(take,dtype=np.float32)
                    answer = titanic_sets[np.argmax(model_prediction.forward_propagation(takem))]



                else:
                    print("Wrong Input: ")
                    time.sleep(1)  
               
            except FileNotFoundError as e:
                print(e)
                
        case 4:
            os.system('cls')
            ask = int(input("Press 1 for Image file deletion\nPress 2 for Titanic Prediction file deletion"))
            
            try:
                if ask == 1 :
                    ask = int(input("Do you really want to delete Image file(1/0) : "))
                    if ask == 1:
                        loading_screen()
                        model_image.forget()
                        continue
                elif ask == 2:
                    ask = int(input("Do you really want to delete Prediction file(1/0) : "))
                    if ask == 1:
                        loading_screen()
                        model_prediction.forget()
                        continue
                else :
                    print("Wrong Input: ")
                    time.sleep(1)
                 
            except:
                 print("Enter a number !!!!!")

            
        
        case 5:
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
    os.system('cls')

