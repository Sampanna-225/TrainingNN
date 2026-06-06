import gzip
import numpy as np
import os

dir = os.path.dirname(os.path.abspath(__file__))

paths= {
    "img_path" : os.path.join(dir, "train-images-idx3-ubyte.gz"),
    "img_test_path" : os.path.join(dir, "t10k-images-idx3-ubyte.gz"),
    "label_path" : os.path.join(dir, "train-labels-idx1-ubyte.gz"),
    "label_test_path" : os.path.join(dir, "t10k-labels-idx1-ubyte.gz"),
    "titanic_x_path" : os.path.join(dir,"titanic.csv")
}

for key, filept in paths.items():
    if not os.path.exists(filept):
        raise FileNotFoundError("All files are not intact consider downloading all the files.")

def unzip_image(target:str) -> np.ndarray:
    """
    MINST has 16 byte header ignore it using offset.
    -1 tells the reshape to do the math for you counting each row taht consists of 784 datas.
    A data of a pixel specially at unit8 gives value 0 to 255 by dividing by 255 we normalize the data to either 0 or 1.
    """
    with gzip.open(target,'rb') as f:
        X_data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return X_data.reshape(-1,784)/255.0
    
    
def unzip_label(target:str) -> np.ndarray:
    """
    MINST has 8 byte header ignore it using offset.
    -1 tells the reshape to do the math for you counting each row taht consists of 784 datas.
    While the label needs indexing we use hot encoding to have 0-9 indexes.
    .shape[0] give number of elements in 1D array
     temp[np.array(sample_num),y_flat] uses advance indexing where the lets take np.arange(3) gives 1D array [0,1,2] and yflat be [2,5,4] the paring begins as (0,2),(1,5),(3,4) so =1 adds 1 on the specific index.
    """
    with gzip.open(target,"rb") as f:
        Y_data = np.frombuffer(f.read(), dtype=np.uint8 , offset=8)
        y_flat = Y_data.flatten()
        sample_num = y_flat.shape[0]
        temp = np.zeros((sample_num,10)) #2D
        temp[np.arange(sample_num),y_flat] = 1
        return temp
    
def min_max(li:list) -> list:
    max0 = max(li)
    min0 = min(li)
    flist =[]
    for i in li:
        ans = (i-min0)/(max0-min0)
        flist.append(ans)
    return flist

def extract_csv(target:str) -> np.ndarray:
    with open(target,mode="r",encoding='utf-8') as f:
        #reads the first line to skip the header
        header = f.readline()
        Input_M =[]
        survived= []
        age_list =[]
        fare_list = []
        for line in f:
            line = line.strip() # removes whitespaces and \n lines
            if not line:
                continue #skips 
            
            column = line.split(',') # arranges features into list that are seperated by ,
            try:
                survived.append(int(column[0]))
                pclass = column[1]
                sex = 0.0 if column[2].lower() == "male" else 1.0
                age = float(column[3]) if column[3] else 28.0
                age_list.append(age)
                fare = float(column[6]) if column[6] else 14.45
                fare_list.append(fare)
                pclass_encoding = [0.0,0.0,0.0]
                pclass_encoding[int(pclass)-1] = 1.0

                
                Input_M.append([pclass_encoding[0],pclass_encoding[1],pclass_encoding[2],sex,age,fare])
            except (ValueError,IndexError):
                continue
        
        mimx1 = min_max(age_list)
        mimx2 = min_max(fare_list)
        X = np.array(Input_M,dtype=np.float32)

        X[:,4] = mimx1
        X[:,5] = mimx2


        survived_y = np.zeros((len(survived),2)) #-> it expects a tuple hence the ()
        survived_y[np.arange(len(survived)),survived] = 1

        
        Y = np.array(survived_y,dtype=np.float32)

        return X , Y , age_list , fare_list

X_titanic , Y_titanic , a ,f = extract_csv(paths["titanic_x_path"])



X_image = unzip_image(paths["img_path"])
X_image_test = unzip_image(paths["img_test_path"])

Y_label = unzip_label(paths["label_path"])
y_label_test = unzip_label(paths["label_test_path"])