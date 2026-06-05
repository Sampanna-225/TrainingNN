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

    croped = res_up[y:y+h, x:x+w]
    
    max_dim = max(h, w) #Selects the highest dimension of the cropped peice so that the padding fits all the data selected
    
    scalling = 20.0 / max_dim

    h_new , w_new = max(1,int(h* scalling)),max(1,int(w* scalling)) #int to elimiate teh decimals as pixels dimensions are in the form of +ve integers and avoid zero dimentions.

    resize = cv2.resize(croped, (w_new,h_new), interpolation=cv2.INTER_AREA) # near 20 padding to compensate the padding

    empty_cavas = np.zeros((28,28),dtype=np.uint8)
    M = cv2.moments(resize)
    if M["m00"] != 0:
        cx , cy = M["m10"]/M["m00"] , M["m01"]/M["m00"]
    else:
        cx , cy = w_new / 2 , h_new / 2 #get the middle coordinates if the above condition fails.

    #Finding the start if the top corner coodinates
    start_x = int(14-cx)
    start_y = int(14-cy)

    #Determining the slice limit for empty canvas:
    x1 , x2 = max(0,start_x) , min(28,start_x+w_new)
    y1 , y2 = max(0,start_y) , min(28,start_y+h_new) 

    #Determining the slice limit for the source:
    #Handles overflows:
    #finds the start of the source from origin. If the start of canvas is greater than 0 than start of source is taken fully from start (0) of the respected axis. If the start of the canvas is placed in negatives the source is cropped towards the positive start of the canvas to secure the orgin of the canvas so the image doesnt overflow.
    source_x1 = max(0,-start_x) 
    source_y1 = max(0,-start_y)
    #takes distance from the start to the respected axis difference hense the dimensions
    source_x2 = source_x1 + (x2-x1) 
    source_y2 = source_y1 + (y2-y1)


    empty_cavas[y1:y2 , x1:x2] = resize[source_y1:source_y2, source_x1:source_x2]# y means the rows and x means the columns 
    # the differene dely and delx is taken in resize to fit exact amount of data from 0 to length of empty canvas.

    normalize = empty_cavas.astype(np.float64)/255.0

    return normalize.reshape(1,784) # production format