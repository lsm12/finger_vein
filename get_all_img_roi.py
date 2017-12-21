#!/usr/bin/python
import numpy as np
import os
import cv2

def pathFile(path):
    return os.getcwd() + '/' + path

def brightestColumn(img):
    w, h = img.shape
    r = range(h / 2, h - 1)
    c = range(0, w - 1)
    return img[c][:,r].sum(axis=0).argmax()

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

def getRoiHCut2(img, p0):
    h, w = img.shape

    maxTop = np.argmax(img[0: h / 2, 0])
    minTop = np.argmax(img[0: h / 2, w-1])

    maxBottom = np.argmax(img[(3 * h / 4): h, 0]) + 3 * h / 4
    minBottom = np.argmax(img[(3 * h / 4): h, w-1]) + 3 * h / 4

    maxTop = (2*maxTop + minTop) / 3
    maxBottom = (maxBottom + 2*minBottom) / 3

    return img[maxTop:maxBottom,:]

def getRoi(img):
    height, width = img.shape
    heightDist = height / 4

    w = img.copy()
    w1 = w[heightDist:3 * heightDist,width / 4:]
    p0 = brightestColumn(w1) + heightDist + height / 2
    pCol = w[:,p0:p0 + 1]

    pColInv = pCol[::-1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    w1_2 = clahe.apply(w[:, (p0 / 20):(p0 + p0 / 2)])
    w2 = getRoiHCut2(w1_2, p0)

    res = cv2.resize(w2, (180, 100), interpolation=cv2.INTER_CUBIC)

    return clahe.apply(res)

def logImg(img):
    return img.astype(float) / 255


imgs = []
dbDir = os.getcwd() + "/db/"
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

p0Imgs = [i.replace('db', 'gab_roi_db') for i in imgs]

filters = build_filters()
for index, imgPath  in enumerate(imgs):
    img = cv2.imread(imgPath + ".bmp", 0)
    res0 = process(getRoi(img), filters)
    cv2.imwrite(p0Imgs[index] + ".png", res0)
    print index


cv2.waitKey(0)
cv2.destroyAllWindows()