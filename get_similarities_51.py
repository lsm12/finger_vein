#!/usr/bin/python
import numpy as np
import os
import cv2
from random import shuffle
import json


def getAverage(img, n):
    """img as a square matrix of numbers"""
    s = np.sum(img)
    n2 = 2 * n + 1
    return float(s)/ (n2 * n2)


def getStandardDeviation(img, n):
    avg = getAverage(img, n)
    imAvg = img - avg
    imAvg *= imAvg
    s = np.sum(imAvg)
    return (s**0.5) / (2 * n + 1)


def zncc(img1, img2, n):
    stdDeviation1 = getStandardDeviation(img1, n)
    stdDeviation2 = getStandardDeviation(img2, n)
    avg1 = getAverage(img1, n)
    avg2 = getAverage(img2, n)

    nImg1 = img1 - avg1
    nImg2 = img2 - avg2
    nImg1 *= nImg2
    s = np.sum(nImg1)
    return float(s) / ((2 * n + 1)**2 * stdDeviation1 * stdDeviation2)


def getPositions(windowSize):
    x = []
    y = []
    idx = 0
    while idx < 179:
        a = 0 if idx == 0 else 8
        idx = (idx - a if idx - a != 129 else idx - a - 1)
        x.append(idx)

        idxF = idx + windowSize
        idx = idxF

    idx = 0

    while idx < 99:
        a = 0 if idx == 0 else 3
        idx -= a
        y.append(idx)
        idxF = idx + windowSize
        idx = idxF
    return x, y


def cutImage(img, posY, posX, windowSize):
    imgCut = []

    for i in range(len(posX)):
        x = posX[i]
        for j in range(len(posY)):
            y = posY[j]
            imgCut.append(img[x:x + windowSize, y:y + windowSize])
    return imgCut


def getZmnnDist(A, B):
    windowSize = 51
    x, y = getPositions(windowSize)

    aCut = cutImage(A, x, y, windowSize)
    bCut = cutImage(B, x, y, windowSize)
    summ = 0
    for i in range(len(aCut)):
        summ += zncc(aCut[i], bCut[i], 26)
    return summ / 8.0;


def getClass(fileName):
    fingers = {
        'index': 0,
        'middle': 1,
        'ring': 2
    }

    hands = {
        'left': 3,
        'right': 4
    }
    f = fileName.split("/")[-3:]
    matchingClass = int(f[0]) * 100 + int(hands[f[1]]) * 10 + fingers[f[2].split('_')[0]]
    return matchingClass


def getImg(imageName, dic):
    imgClass = getClass(imageName)
    if imgClass not in dic:
        dic[imgClass] = {}
        dic[imgClass][imageName[-1]] = cv2.imread(imageName + ".png", 0)
    elif imageName[-1] not in dic[imgClass]:
        dic[imgClass][imageName[-1]] = cv2.imread(imageName + ".png", 0)

    image = dic[imgClass][imageName[-1]]
    return image, dic


def maxArray(dists, distClass, kVal, kClass):
    for idx in range(len(dists)):
        if kVal > dists[idx]:
            val = dists[idx]
            nClass = distClass[idx]
            dists[idx] = kVal
            distClass[idx] = kClass

            dists, distClass = maxArray(dists, distClass, val, nClass)
            break

    return dists, distClass


def getGreatestClass(imgA, closClasses, classDistRam):
    aClass = str(getClass(imgA))
    aNum = str(imgA.split("_")[-1])
    weight = {}
    maxDist = 0
    maxClass = 0

    for bClass in closClasses:
        summ  = 0
        count = 0
        for bNum in range(1, 7):
            if (str(aClass) + str(bClass)) in classDistRam:
                summ += classDistRam[(str(aClass) + str(bClass))]
                count += 1
            elif (str(bClass) + str(aClass)) in classDistRam:
                summ += classDistRam[str(bClass) + str(aClass)]
                count += 1
        mean = summ / float(count if count > 0 else 1.0)
        if mean > maxDist:
            maxDist = mean
            maxClass = bClass
    return maxClass, maxDist


def getDist(classDistRam, A, aName, B, bName):
    aClass = str(getClass(aName))
    aNum = str(aName.split("_")[-1])
    bClass = str(getClass(bName))
    bNum = str(bName.split("_")[-1])
    if (aClass + "-" + aNum + "_" + bClass + bNum) in classDistRam:
        dist = classDistRam[(aClass + bClass)]
    elif (bClass + "-" + bNum + "_" + aClass + "-" + aNum) in classDistRam:
        dist = classDistRam[bClass + "-" + bNum + "_" + aClass + "-" + aNum]
    else:
        dist = getZmnnDist(A, B)
        classDistRam[aClass + "-" + aNum + "_" + bClass + "-" + bNum] = dist
    return dist, classDistRam


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


threshold = 10
imgRam = {}
classDistRam = {}
k = 5
wrongClassification = []
rightClassification = []

for i in range(len(imgs)):
    print i
    A, imgRam = getImg(imgs[i], imgRam)
    nameA = imgs[i].split("/")[-3:]
    aClass = getClass(imgs[i])

    dists = []
    kClasses = []
    for j in range(len(imgs)):
        if i == j:
            continue
        B, imgRam = getImg(imgs[j], imgRam)
        bClass = getClass(imgs[j])
        bNum = imgs[j].split("_")[1]

        dst, classDistRam = getDist(classDistRam, A, imgs[i], B, imgs[j])

        dists.append(dst)
        kClasses.append(bClass)
        nameB = imgs[j].split("/")[-3:]
    sortedVal = [(x, y) for x, y in sorted(zip(dists,kClasses), key=lambda pair: pair[0])]
    closClasses = [y[1] for y in sortedVal[-k:]]
    predictedClass, wVal = getGreatestClass(imgs[i], closClasses, classDistRam)
    closValues = [y[0] for y in sortedVal[-k:]]

    if predictedClass == aClass:
        if closValues[-1] < threshold:
            rejectedWrong.append((predictedClass, aClass))
            continue
        rightClassification.append((predictedClass, aClass))
    else:
        if closValues[-1] < threshold:
            rejectedCorrect.append((predictedClass, aClass))
            continue
        wrongClassification.append((predictedClass, aClass))

with open('window_51_gab.json', 'w') as fp:
    json.dump(classDistRam, fp)