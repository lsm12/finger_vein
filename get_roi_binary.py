#!/usr/bin/python
import numpy as np
import os
import cv2

def pathFile(path):
    return os.getcwd() + '/' + path

def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        
        params = {'ksize':(ksize, ksize), 'sigma':3.3, 'theta':theta, 'lambd':18.3,
                  'gamma':4.5, 'psi':0.89, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def Gk(img, x, y):
    # for 
    scale = 16
    first = 1 / np.sqrt(2 * np.pi * scale * scale)
    ePart = np.exp((x * x + y * y) / (2 * scale * scale))
    g = first * ePart

    return g

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned

def logImg(img):
    return img.astype(float) / 255


imgs = []
dbDir = os.getcwd() + "/gab_roi_db/"
people = os.listdir(dbDir)
people.sort()

for person in people:
    personDir = dbDir + person + "/"
    hands = os.listdir(personDir)

    for hand in hands:
        handDir = personDir + hand + "/"
        mg = os.listdir(handDir)
        mg.sort()
        imgs = imgs + [handDir + s.split(".")[0] for s in mg if not s.split(".")[0] == "Thumbs"]

p0Imgs = [i.replace('gab_roi_db', 'gab_bin_db') for i in imgs]

for index, imgPath  in enumerate(imgs):
    img = cv2.imread(imgPath + ".png", 0)

    ''' BINARIZATION '''
    b = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    b1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)

    ''' ERODE '''
    kernel = np.ones((2,7),np.uint8)
    opening = cv2.morphologyEx(b1, cv2.MORPH_OPEN, kernel)

    res1 = cv2.normalize(cv2.bitwise_not(opening) - b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    res0 = zhangSuen(res1)
    res0 = cv2.normalize(res0, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print(index)
    cv2.imwrite(p0Imgs[index] + ".png", res0)
    # break
