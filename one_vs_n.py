#!/usr/bin/python
import numpy as np
import os
import cv2
from random import shuffle
import json
import sys


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
        bClass = str(bClass)
        summ  = 0
        count = 0
        for bNum in range(1, 7):
            bNum = str(bNum)
            if (aClass + "-" + aNum + "_" + bClass + "-" + bNum) in classDistRam:
                summ += classDistRam[(aClass + "-" + aNum + "_" + bClass + "-" + bNum)]
                count += 1
            elif (bClass + "-" + bNum + "_" + aClass + "-" + aNum) in classDistRam:
                summ += classDistRam[bClass + "-" + bNum + "_" + aClass + "-" + aNum]
                count += 1
        mean = summ / float(count if count > 0 else 1.0)
        if mean > maxDist:
            maxDist = mean
            maxClass = bClass
    return maxClass, maxDist


def getDist(classDistRam, aName, bName):
    aClass = str(getClass(aName))
    aNum = str(aName.split("_")[-1])
    bClass = str(getClass(bName))
    bNum = str(bName.split("_")[-1])
    if (aClass + "-" + aNum + "_" + bClass + "-" + bNum) in classDistRam:
        dist = classDistRam[(aClass + "-" + aNum + "_" + bClass + "-" + bNum)]
    elif (bClass + "-" + bNum + "_" + aClass + "-" + aNum) in classDistRam:
        dist = classDistRam[bClass + "-" + bNum + "_" + aClass + "-" + aNum]

    return dist, classDistRam


imgs = []
dbDir = os.getcwd() + "/roi_db/"
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


thresholds = np.arange(0.0, 1.05, 0.05)
# print thresholds
# quit()
classDistRam = {}
with open("window_51_gab.json") as tweetfile:
    classDistRam = json.load(tweetfile)
imgRam = {}

k = 5
threshold = 0.75

rightPred = 0
wrongPred = 0
rejected = 0

for i in range(len(imgs)):
    nameA = imgs[i].split("/")[-3:]
    aClass = getClass(imgs[i])

    dists = []
    kClasses = []
    for j in range(len(imgs)):
        if i == j:
            continue

        bClass = getClass(imgs[j])
        bNum = imgs[j].split("_")[1]

        if aClass != bClass:
            continue

        dist, _ = getDist(classDistRam, imgs[i], imgs[j])

        dists.append(dist)
        kClasses.append(bClass)
    sortedVal = [(x, y) for x, y in sorted(zip(dists,kClasses), key=lambda pair: pair[0])]
    closClasses = [y[1] for y in sortedVal[-k:]]
    predictedClass, wVal = getGreatestClass(imgs[i], closClasses, classDistRam)
    closValues = [y[0] for y in sortedVal[-k:]]

    if closValues[-1] < threshold:
        rejected += 1
        print("rejected", predictedClass, aClass)
    elif str(predictedClass) == str(aClass):
        rightPred += 1
        print("correct", predictedClass, aClass)
    else:
        wrongPred += 1
        print("wrong", predictedClass, aClass)

print("Total right predictions: %d" % (rightPred))
print("Total wrong predictions: %d" % (wrongPred))
print("Total rejected: %d" % (rejected))