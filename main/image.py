import cv2
import numpy as np


def image_converter(pathname:str) -> np.ndarray: 
    grey_scale = cv2.imread(pathname, cv2.IMREAD_GRAYSCALE)
    if grey_scale is None:
        raise FileNotFoundError("The image file is not loaded or corrpted")
    
    img_inverse = 255 - grey_scale
    res_up =cv2.threshold(img_inverse,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #--> touple index 1 = image data
    #converts any pixels above OTSU threshold to be white (255)

    cords = cv2.findNonZero(res_up)    # --> 3D matrix coordinates of all white pixels

    x, y, w, h = cv2.boundingRect(cords) # Creates a rectangle that covers every coordinate given by cords
    # x is x coordinate of top left
    # y is y coordinate of top left
    # w is width of rectangle 
    # h is height of rectangle 
    #(x, y) Top-Left
    #      +----------------------+
        #   |      * *** 
        #   |    ***** ***** |  
        #   |     ********* |  h (Height)
        #   |       **** |
        #   +----------------------+
        #          w (Width)

    croped = res_up[y:y-h, x:x+h]
    

   
    resize = cv2.resize(res_up, (20,20), interpolation=cv2.INTER_AREA) # 20 to compensate the padding

    h ,w = resize.shape
    h_pad , w_pad = 4, 4# 4 pixels of padding on all ends so the final becomes (20+8,20+8)
    padded = cv2.copyMakeBorder(resize, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=0)#padds the images with value=0(black)

    normalize = padded.astype(np.float64)/255.0

    return normalize.reshape(-1,784) # production fromat