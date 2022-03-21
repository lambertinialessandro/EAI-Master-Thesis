# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:17:53 2022

@author: lambe
"""

"""
r11 r12 r13 tx
r21 r22 r23 ty
r31 r32 r33 tz
0   0   0   1
is represented in the file as a single row:
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
"""

import os
import math
import time
import numpy as np
from enum import Enum

import cv2
from PIL import Image
from scipy.ndimage import grey_dilation, grey_erosion
from skimage.morphology import disk
from skimage.filters.rank import entropy

from params import FLAG_DEBUG_PRINT, FLAG_INFO_PRINT, \
        dir_main, dir_Dataset, dir_Model, dir_History, path_sequences, path_poses,\
        img_size
from utility import PrintManager, bcolors

pm = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT)

WIDTH = img_size[0] # img_size[0], 1280
HEIGHT = img_size[1] # img_size[1], 384
CHANNELS = 3

class EnumPreproc(Enum):
    UNCHANGED = "0"
    CROPPING = "1"

    QUAD_CANNY_ENTOPY_DILATED = "100"

    def genFun(self):
        if self == EnumPreproc.UNCHANGED:
            def f(imgPath, imgSize):
                im = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
                im = cv2.resize(im, imgSize, interpolation = cv2.INTER_AREA);
                return im
            return f
        elif self == EnumPreproc.CROPPING:
            def f(imgPath, imgSize):
                im = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
                w, h, _ = im.shape
                w2 = round(w/2)

                top = h-imgSize(1)
                bottom = h

                is_w2 = round(imgSize/2)
                left = w2-is_w2
                if imgSize%2 == 0:
                    right = w2+is_w2
                else:
                    right = w2+is_w2+1

                im = im[top:bottom, left:right];
                return im
            return f
        elif self == EnumPreproc.QUAD_CANNY_ENTOPY_DILATED:
            def f(imgPath, imgSize):
                imRGB = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
                imRGB = cv2.resize(imRGB, (1241, 376), interpolation = cv2.INTER_AREA);
                imGray = cv2.cvtColor(imRGB, cv2.COLOR_BGR2GRAY)

                canny_img = cv2.Canny(np.array(imRGB), 100, 200)
                canny_img = Image.fromarray(canny_img)

                scaled_entropy = canny_img / np.max(canny_img)
                entropy_image = entropy(scaled_entropy, disk(2))
                scaled_entropy = entropy_image / entropy_image.max()
                mask = scaled_entropy > 0.75
                maskedImg = imGray * mask

                dilated = grey_dilation(maskedImg, footprint=np.ones((3,3)))
                dilated = grey_erosion(dilated, size=(3,3))
                dilated = np.array(Image.fromarray(dilated))

                im_CED = np.reshape(dilated, (376, 1241, 1))
                quatImg = np.concatenate((im_CED, imRGB), 2)
                quatImg = cv2.resize(quatImg, imgSize, interpolation = cv2.INTER_AREA);
                return quatImg
            return f

typePreprocessing = EnumPreproc.UNCHANGED
processImg = typePreprocessing.genFun()
typePreprocessingQuat = EnumPreproc.QUAD_CANNY_ENTOPY_DILATED
processImgQuat = typePreprocessingQuat.genFun()

def checkExistDirs(dirs):
    for dir in dirs:
      if not os.path.exists(dir):
        os.makedirs(dir)
        pm.printD(dir + " --> CREATED")
      else:
        pm.printD(dir + " --> ALREADY EXIST")

def readImgsToList(imagesSet, N, path, files, processImgF, imgSize):
    pos = 0
    img1 = []
    img2 = []

    for f in files:
        pm.printProgressBarI(pos, N)
        img2 = processImgF(path+f, img_size)

        if pos > 0:
            img = np.concatenate([img1, img2], axis=-1)
            imagesSet.append(img)

        img1 = img2
        pos += 1

    pm.printProgressBarI(N, N)
    imagesSet = np.reshape(imagesSet, (-1, )+imgSize)

def isRotationMatrix(R):
    RT = np.transpose(R)
    n = np.linalg.norm(np.identity(3, dtype = R.dtype) - np.dot(RT, R))
    return n < 1e-6

def rotationMatrix2EulerAngles(R):
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    if  sy < 1e-6:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    else:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])

    return np.array([x, y, z])

def poseFile2poseRobot(posef):
    p = np.array([posef[3], posef[7], posef[11]])
    R = np.array([[posef[0], posef[1], posef[2]],
                  [posef[4], posef[5], posef[6]],
                  [posef[8], posef[9], posef[10]]])

    angles = rotationMatrix2EulerAngles(R)
    pose = np.concatenate((p, angles))
    return pose

def readPosesFromFile(posesSet, N, path):
    pose1 = []
    pose2 = []

    with open(path, 'r') as f:
        for pos in range(N):
            pm.printProgressBarI(pos, N)
            posef = np.fromstring(f.readline(), dtype=float, sep=' ')
            pose2 = poseFile2poseRobot(posef)

            if pos > 0:
                pose = pose2-pose1
                posesSet.append(pose)

            pose1 = pose2
        pm.printProgressBarI(N, N)

def convertDataset():
    suffix = "_"+str(WIDTH)+"_"+str(HEIGHT)+"_loaded.npy"
    suffixQuat = "_"+str(WIDTH)+"_"+str(HEIGHT)+"_Quat_loaded.npy"
    for dirSeqName in os.listdir(path_sequences):
        dirSeq = path_sequences+dirSeqName+"/"
        if not os.path.isdir(dirSeq):
            continue

        pm.printI(bcolors.DARKYELLOW+"Converting: "+dirSeqName+bcolors.ENDC)
        for imgsSeqName in ['image_2']:#os.listdir(dirSeq):
            imgsSeq = dirSeq+imgsSeqName+"/"
            if not os.path.isdir(imgsSeq):
                continue

            x_files = sorted(os.listdir(imgsSeq))
            imgs_N = len(x_files)

            if os.path.isfile(dirSeq+imgsSeqName+suffix):
                pm.printD("Already converted ["+dirSeqName+"/"+imgsSeqName+"]!!")
            else:
                pm.printD("Converting --> ["+dirSeqName+"/"+imgsSeqName+"]")
                initT = time.time()

                imagesSet = []
                readImgsToList(imagesSet, imgs_N, imgsSeq, x_files, processImg,
                               (CHANNELS, WIDTH, HEIGHT))
                pm.printD("Saving on file: "+dirSeq+imgsSeqName+suffix)
                np.save(dirSeq+imgsSeqName+suffix, imagesSet, allow_pickle=False)
                elapsedT = time.time() - initT
                pm.printD("Time needed: %.2fs for %d images"%(elapsedT, imgs_N))

            pm.printI(bcolors.DARKGREEN+"Done: "+dirSeq+imgsSeqName+suffix+bcolors.ENDC)

            if os.path.isfile(dirSeq+imgsSeqName+suffixQuat):
                pm.printD("Already converted ["+dirSeqName+"/"+imgsSeqName+"_Quat]!!")
            else:
                pm.printD("Converting --> ["+dirSeqName+"/"+imgsSeqName+"_Quat]")
                initT = time.time()

                imagesSet = []
                readImgsToList(imagesSet, imgs_N, imgsSeq, x_files, processImgQuat,
                               (CHANNELS+2, WIDTH, HEIGHT))
                pm.printD("Saving on file: "+dirSeq+imgsSeqName+suffixQuat)
                np.save(dirSeq+imgsSeqName+suffixQuat, imagesSet, allow_pickle=False)
                elapsedT = time.time() - initT
                pm.printD("Time needed: %.2fs for %d images"%(elapsedT, imgs_N))

            pm.printI(bcolors.DARKGREEN+"Done: "+dirSeq+imgsSeqName+suffixQuat+bcolors.ENDC)

        poseFileName = path_poses+dirSeqName+"_pose_loaded.npy"
        if os.path.isfile(poseFileName):
            pm.printD("Already converted [poses/"+dirSeqName+".txt]!!")
            pm.printI(bcolors.DARKGREEN+"Done: "+poseFileName+bcolors.ENDC)
        else:
            pm.printD("Converting --> [poses/"+dirSeqName+".txt]")
            initT = time.time()

            posesSet = []
            fileName = path_poses+dirSeqName+'.txt'
            if os.path.isfile(path_poses+dirSeqName+'.txt'):
                readPosesFromFile(posesSet, imgs_N, fileName)

                pm.printD("Saving on file: "+poseFileName)
                np.save(poseFileName, posesSet, allow_pickle=False)
                elapsedT = time.time() - initT
                pm.printD("Time needed: %.2fs for %d poses"%(elapsedT, imgs_N))

                pm.printI(bcolors.DARKGREEN+"Done: "+poseFileName+bcolors.ENDC)
            else:
                pm.printD(bcolors.WARNING+fileName+" does not exists!!"+bcolors.ENDC)



def main():
    pm.printI(bcolors.DARKYELLOW+"Checking directories"+bcolors.ENDC+" ###\n")
    dirs = [dir_main, dir_Dataset, path_sequences, dir_Model, dir_History]
    checkExistDirs(dirs)
    pm.printI("Directories checked!\n")

    pm.printI(bcolors.DARKYELLOW+"Converting dataset"+bcolors.ENDC+" ###")
    convertDataset()
    pm.printI("Done dataset convertion!\n")

if __name__ == "__main__":
    main()











