import numpy as np
import os
import matplotlib.pyplot as plt #pyplot helps run plot function
from typing import Optional,List
from tqdm import tqdm

def sigmoid(x): #to get non linear output
    return 1.0/(1.0 + np.exp(-np.clip(x,-500,500)))# clip max and min 

def sigmoid_derivative(x): #to get the derivative of sigmoid function
    return x*(1-x) # expected value of sigmoid function 

def RelU(x):
    return np.where(x>0 , x , x * 0.01)

def RelU_grad(z):
    return np.where(z>0 , 1 , 0.01)

#Functions like input layer --> hidden layer --> output layer
# Class creation for training

class NeuralBrain:
    """
    A class that acts as a Neural Network bias and tools of viewing or managing it 

    Handles training, forward/backward propagation, and weight persistence
    using NumPy and file-system modules.
    As the input and hidden layers grows the unscaled weights give out massive negative or positive number causing gradient to be flatlined. So the square root is scaled with the number of layers to impact the weights 

    """
    def __init__(self,x,y,input_size,hidden_size,output_size,lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x = x
        self.y = y
        self.lr = lr
        self.mini_batch_shape = self.x.shape[0] if self.x < 32 else 32
        self.loss_history=[] 
        self.dir_path = os.path.dirname(os.path.abspath(__file__)) # get absolute path of the folder to brain
        self.project_root = os.path.dirname(self.dir_path) # sets the directry one level up to initial file
        self.data_folder = os.path.join(self.project_root,"data")
        self.file_path = os.path.basename(__file__)# file path

        if self.input_size >= 784:
            self.file_name = os.path.join(self.data_folder,f"{os.path.splitext(self.file_path)[0]}_image.npz")# remove extension
        else:
            self.file_name = os.path.join(self.data_folder,f"{os.path.splitext(self.file_path)[0]}_tabular.npz")

        # takes random value for weight of range of input layer to hidden layer
        self.__W1 = np.random.randn(self.input_size,self.hidden_size)* np.sqrt(2.0 / self.input_size)#randn gives +ve to -ve value
        # takes bias value for hidden layer
        self.__B1 = np.zeros((1,self.hidden_size))# 1 to avoid shape mismatch
                                                #zeros as starting 0 is inital
        # takes random value for weight of range of hidden layer to output layer
        self.__W2 = np.random.randn(self.hidden_size,self.output_size)*np.sqrt(2.0 / self.hidden_size)
        # takes bias value for output layer
        self.__B2 = np.zeros((1,self.output_size))# 1 to avoid shape mismatch
                                                 #zeros as starting 0 is inital
                        
    
    def forward_propagation(self, x1:np.ndarray) -> np.ndarray:
        """
        Standard process of forward biasing with equation x.w + b to alter data each layer

        Args:
            x (np.ndarray): The input feature matrix.
            
        Returns:
            np.ndarray: The activated output of the final layer.
        """
        self.cx = x1
        # from input to hidden tinkering with weight and bias
        # directly passed x helps reusability
        self.z1 = np.dot(x1,self.__W1) + self.__B1 #Main concept == x.__W1 + __B1 __> output of hidden layer
        # self.a1 = sigmoid(self.z1) #activation function
        self.a1 = RelU(self.z1)

        # from hidden to output tinkering with weight and bias
        self.z2 = np.dot(self.a1,self.__W2) + self.__B2 #Main concept == a1.__W2 + __B2
        # self.a2 = sigmoid(self.z2) #activation function
        self.a2 = self.softmax()
        return self.a2 #final output of forward propagation

    def backward_propagation(self, output:np.ndarray,batching_y:np.ndarray) -> None:
        """
        backwards checking with error gradient wise change.

        Args:
            x (np.ndarray): The input feature matrix
        
        Return:
            None
        
        We use gradient decent -= rather than += becuase we want to oppose the direction for the error gradient to make it as mininmum error as possible.

        """
        n = self.mini_batch_shape
        # from output to hidden layer CROSS ENTROPY
        self.error_slope = output - batching_y #to know if the output obtained is greater or less than the required output.  
        # self.error_slope = self.error * sigmoid_derivative(output) #hybrid architecture
        # self.error_slope = sigmoid_derivative(output)*self.error

    

        #from hidden to input layer
        self.hidden_error = self.error_slope.dot(self.__W2.T) #==> matrix 3*2 . 2*3 == 3*3
        self.hidden_error_slope = self.hidden_error * RelU_grad(self.z1)
        # self.hidden_error_slope = sigmoid_derivative(self.a1)*self.hidden_error

        #correcting values form output to hidden layer
        self.__W2 -= self.a1.T.dot(self.error_slope)*(self.lr/n) # matrix multiplication dimensions. => based on value they recieve so hidden output
        self.__B2 -= np.sum(self.error_slope,axis=0,keepdims=True)*(self.lr/n) #axis = 0 collapising rows
        
        #correcting calues from hidden to outpu layer
        self.__W1 -= self.cx.T.dot(self.hidden_error_slope)*(self.lr/n) # based on value they recieve so x
        self.__B1 -= np.sum(self.hidden_error_slope,axis=0,keepdims=True)*(self.lr/n)

    def train(self,process:int) -> None:
        """
        Trains the data and appends the loss with each successive epoch.

        Args:
            int 
        
        Return:
            None
        """
        for epoch in tqdm(range(process)):

            index = np.arange(self.x.shape[0]) # creates a 1D array of element 0 to rows of self.x
            np.random.shuffle(index) # suffles the indexes so the machine doesnt end up memorizing

            X_shuffled = self.x[index] #creates a view of metadata of the array of the row index 0 to shape[0]; hence pointer is stored to optimize the loops.
            Y_shuffled = self.y[index]

            for i in range(0,self.x.shape[0],32): # in range from 0 to excluisve rows of x in step 32
                Batch_x = X_shuffled[i:i+32] # only row indexing as the indexing of a 2D matrix is of A[x,y]
                Batch_y = Y_shuffled[i:i+32]
                self.mini_batch_shape = Batch_x.shape[0]

                b = self.forward_propagation(Batch_x) # mini packets forward bias

                self.backward_propagation(b,Batch_y) # backward bias of mini packets

            if epoch % 100 == 0 :
                check = self.forward_propagation(self.x) # due to mini batching passing 32 sliced batch , we check by passing a full forward propagation to accumulate the loss of the whole data.
                loss = - np.sum(self.y*np.log(check+1e-16))/self.x.shape[0] # to calculate the loss of cross entropy process. the low decimal point is to avoid log(0)
                self.loss_history.append(loss) # to know learning rate
                if len(self.loss_history) >=10:
                    if max(self.loss_history[-10: ]) - min(self.loss_history[-10: ]) <1e-11:
                        return #Float Point Precision tuning
                    
                # print(f"Loss {epoch} = {loss:.5f}")

    def softmax(self) -> np.ndarray: #softmax function that helps calculate the certainty e^x/sum(e^x)
        self.numerator = np.exp(self.z2-np.max(self.z2, axis=-1 , keepdims=True))# Max subtraction added to prevent floating-point infinity overflow
        self.denomenator = np.sum(self.numerator,axis=-1, keepdims=True)
        return (self.numerator/self.denomenator) 
    
    def learn(self) -> None:
        np.savez_compressed(self.file_name,w1=self.__W1,w2=self.__W2,b1=self.__B1,b2 = self.__B2)
        print("Saved successfully")
        print(self.loss_history)

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
    model.forward_propagation(x)
    # print(model.softmax())
    model.graphplot()