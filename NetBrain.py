import numpy as np
import matplotlib.pyplot as plt #pyplot helps run plot function

def sigmoid(x): #to get non linear output
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x): #to get the derivative of sigmoid function
    return x*(1-x) # expected value of sigmoid function 


#Functions like input layer --> hidden layer --> output layer
# Class creation for training
class NeuralBrain:
    def __init__(self,x,y,input_size,hidden_size,output_size,lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x = x
        self.y = y
        self.lr = lr
        self.loss_history=[] 

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
                        
    
    def forward_propagation(self,x):
        # from input to hidden tinkering with weight and bias
        # directly passed x helps reusability
        self.z1 = np.dot(x,self.__W1) + self.__B1 #Main concept == x.__W1 + __B1 __> output of hidden layer
        self.a1 = sigmoid(self.z1) #activation function

        # from hidden to output tinkering with weight and bias
        self.z2 = np.dot(self.a1,self.__W2) + self.__B2 #Main concept == a1.__W2 + __B2
        self.a2 = sigmoid(self.z2) #activation function
        return self.a2 #final output of forward propagation

    def backward_propagation(self,output):
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

    def train(self,process):
        for epoch in range(process):
            b = self.forward_propagation(self.x)
            self.backward_propagation(b)
            if epoch % 2000 == 0 :
                loss = np.mean((self.y-b)**2)
                self.loss_history.append(loss) # to know learning rate
                print(f"Loss {epoch} = {loss:.4f}")
    
    def graphplot(self):
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