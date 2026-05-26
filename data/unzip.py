import gzip
import numpy as np
import os

dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(dir, "t10k-images-idx3-ubyte.gz")
label_path = os.path.join(dir, "t10k-labels-idx1-ubyte.gz")


def unzip_image() -> np.ndarray:
    """
    MINST has 16 byte header ignore it using offset.
    -1 tells the reshape to do the math for you counting each row taht consists of 784 datas.
    A data of a pixel specially at unit8 gives value 0 to 255 by dividing by 255 we normalize the data to either 0 or 1.
    """
    with gzip.open(img_path,'rb') as f:
        X_data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return X_data.reshape(-1,784)/255
    
def unzip_label() -> np.ndarray:
    """
    MINST has 8 byte header ignore it using offset.
    -1 tells the reshape to do the math for you counting each row taht consists of 784 datas.
    While the label needs indexing we use hot encoding to have 0-9 indexes.
    .shape[0] give number of elements in 1D array
     temp[np.array(sample_num),y_flat] uses advance indexing where the lets take np.arange(3) gives 1D array [0,1,2] and yflat be [2,5,4] the paring begins as (0,2),(1,5),(3,4) so =1 adds 1 on the specific index.
    """
    with gzip.open(label_path,"rb") as f:
        Y_data = np.frombuffer(f.read(), dtype=np.uint8 , offset=8)
        y_flat = Y_data.flatten()
        sample_num = y_flat.shape[0]
        temp = np.zeros((sample_num,10)) #2D
        temp[np.arange(sample_num),y_flat] = 1
        return temp