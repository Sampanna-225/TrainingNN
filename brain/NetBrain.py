import numpy as np
import os
import matplotlib.pyplot as plt #pyplot helps run plot function
from typing import Optional,List
from tqdm import tqdm

def sigmoid(x): #to get non linear output
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x): #to get the derivative of sigmoid function
    return x*(1-x) # expected value of sigmoid function 


#Functions like input layer --> hidden layer --> output layer
# Class creation for training

class NeuralBrain:
    """
    A class that acts as a Neural Network bias and tools of viewing or managing it 

    Handles training, forward/backward propagation, and weight persistence
    using NumPy and file-system modules.

    """
    def __init__(self,x,y,input_size,hidden_size,output_size,lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x = x
        self.y = y
        self.lr = lr
        self.loss_history=[] 
        self.dir_path = os.path.dirname(os.path.abspath(__file__)) # get absolute path of the folder to brain
        self.project_root = os.path.dirname(self.dir_path) # sets the directry one level up to initial file
        self.data_folder = os.path.join(self.project_root,"data")
        self.file_path = os.path.basename(__file__)# file path
        self.file_name = os.path.join(self.data_folder,f"{os.path.splitext(self.file_path)[0]}.npz")# remove extension
        
        # takes random value for weight of range of input layer to hidden layer
        self.__W1 = np.random.rand(self.input_size,self.hidden_size)
        # takes bias value for hidden layer
        self.__B1 = np.zeros((1,self.hidden_size))# 1 to avoid shape mismatch
                                                #zeros as starting 0 is inital
        # takes random value for weight of range of hidden layer to output layer
        self.__W2 = np.random.rand(self.hidden_size,self.output_size)
        # takes bias value for output layer
        self.__B2 = np.zeros((1,self.output_size)) # 1 to avoid shape mismatch
                                                 #zeros as starting 0 is inital
                        
    
    def forward_propagation(self, x:np.ndarray) -> np.ndarray:
        """
        Standard process of forward biasing with equation x.w + b to alter data each layer

        Args:
            x (np.ndarray): The input feature matrix.
            
        Returns:
            np.ndarray: The activated output of the final layer.
        """

        # from input to hidden tinkering with weight and bias
        # directly passed x helps reusability
        self.z1 = np.dot(x,self.__W1) + self.__B1 #Main concept == x.__W1 + __B1 __> output of hidden layer
        self.a1 = sigmoid(self.z1) #activation function

        # from hidden to output tinkering with weight and bias
        self.z2 = np.dot(self.a1,self.__W2) + self.__B2 #Main concept == a1.__W2 + __B2
        self.a2 = sigmoid(self.z2) #activation function
        return self.a2 #final output of forward propagation

    def backward_propagation(self, output:np.ndarray) -> None:
        """
        backwards checking with error gradient wise change.

        Args:
            x (np.ndarray): The input feature matrix
        
        Return:
            None

        """
        # from output to hidden layer
        self.error = self.y - output 
        self.error_slope = sigmoid_derivative(output)*self.error

        #from hidden to output layer
        self.hidden_error = self.error_slope.dot(self.__W2.T) #==> matrix 3*2 . 2*3 == 3*3
        self.hidden_error_slope = sigmoid_derivative(self.a1)*self.hidden_error

        #correcting values form output to hidden layer
        self.__W2 += self.a1.T.dot(self.error_slope)*self.lr # matrix multiplication dimensions. => based on value they recieve so hidden output
        self.__B2 += np.sum(self.error_slope,axis=0,keepdims=True)*self.lr #axis = 0 collapising rows
        
        #correcting calues from hidden to outpu layer
        self.__W1 += self.x.T.dot(self.hidden_error_slope)*self.lr # based on value they recieve so x
        self.__B1 += np.sum(self.hidden_error_slope,axis=0,keepdims=True)*self.lr

    def train(self,process:int) -> None:
        """
        Trains the data and appends the loss with each successive epoch.

        Args:
            int 
        
        Return:
            None
        """
        for epoch in tqdm(range(process)):
            b = self.forward_propagation(self.x)
            self.backward_propagation(b)
            if epoch % 2000 == 0 :
                loss = np.mean((self.y-b)**2)
                self.loss_history.append(loss) # to know learning rate
                if len(self.loss_history) >=10:
                    if max(self.loss_history[-10: ]) - min(self.loss_history[-10: ]) <1e-8:
                        return #Float Point Precision tuning
                    
                # print(f"Loss {epoch} = {loss:.5f}")

    def softmax(self) -> np.ndarray: #softmax function that helps calculate the certainty
        self.numerator = np.exp(self.z2 - np.max(self.z2))
        self.denomenator = np.sum(self.numerator)
        return (self.numerator/self.denomenator)
    
    def learn(self) -> None:
        np.savez_compressed(self.file_name,w1=self.__W1,w2=self.__W2,b1=self.__B1,b2 = self.__B2)
        print("Saved successfully")

    def remember(self) -> None:
        """
        Attempts to load the weights and bias from the save file (.npz)

        Will return "File Maybe Corrupted" if the load hits an error
        """
        try:
            if os.path.exists(self.file_name):
                if os.path.getsize(self.file_name)>0:
                    data = np.load(self.file_name)
                    self.__W1 = data['w1']
                    self.__W2 = data['w2']
                    self.__B1 = data['b1']
                    self.__B2 = data['b2']
            else:
                print("No FILE found !!!")
        except Exception as e: # to view exactly what has happened
            print(f"File Maybe corrupted: ({e})")
    
    def forget(self) -> None:
        """
        Deletes the existing file.
        """
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        else:
            print("No FILE found !!!")
            
    def graphplot(self) -> None:
        """
        Graphical representation of data and loss per epoch.
        """
        if not self.loss_history:
            print("error no loss detected")
            return
        plt.plot(self.loss_history)
        plt.title("Learning graph")
        plt.xlabel('Ephoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    
if __name__ == "__main__":
    x = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

    y = np.array([[0],
                [1],
                [1],
                [0]])

    model = NeuralBrain(x,y,input_size=2,hidden_size=4,output_size=1,lr=0.1)
    model.train(process=100000)
    final = model.forward_propagation(x) 
    print(final)
    model.graphplot()