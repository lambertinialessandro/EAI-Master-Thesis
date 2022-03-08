# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:17:53 2022

@author: lambe
"""

import os
import numpy as np
from enum import Enum
import cv2

from params import FLAG_DEBUG_PRINT, FLAG_INFO_PRINT, \
        dir_main, dir_Dataset, dir_Model, dir_History, path_sequences, path_poses,\
        img_size
from utility import PrintManager, bcolors

pm = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT)

class EnumPreproc(Enum):
    UNCHANGED = "0"
    CROPPING = "1"

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

typePreprocessing = EnumPreproc.UNCHANGED
processImg = typePreprocessing.genFun()

def checkExistDirs(dirs):
    for dir in dirs:
      if not os.path.exists(dir):
        os.makedirs(dir)
        pm.printD(dir + " --> CREATED")
      else:
        pm.printD(dir + " --> ALREADY EXIST")

def readImgsToList(imgs, N, path, files):
    pos = 0
    for f in files:
        pm.printProgressBarI(pos, N)
        imgs[pos, :, :, :] = processImg(path+f, img_size)
        pos += 1
    pm.printProgressBarI(N, N)

def readPosesFromFile(poses, N, path):
    with open(path, 'r') as f:
        for pos in range(N):
            pm.printProgressBarI(pos, N)
            poses[pos, :] = f.readline().split(' ')
        pm.printProgressBarI(N, N)

def convertDataset():
    for dirSeqName in os.listdir(path_sequences):
        dirSeq = path_sequences+dirSeqName+"/"
        if not os.path.isdir(dirSeq):
            continue

        pm.printI(bcolors.OKYELLOW+"Converting: "+dirSeqName+bcolors.ENDC)
        for imgsSeqName in os.listdir(dirSeq):
            imgsSeq = dirSeq+imgsSeqName+"/"
            if not os.path.isdir(imgsSeq):
                continue

            x_files = sorted(os.listdir(imgsSeq))
            imgs_N = len(x_files)

            if os.path.isfile(dirSeq+imgsSeqName+'_loaded.npy'):
                pm.printD("Already converted ["+dirSeqName+"/"+imgsSeqName+"]!!")
            else:
                pm.printD("Converting --> ["+dirSeqName+"/"+imgsSeqName+"]")

                x_imgs = np.empty((imgs_N, img_size[1], img_size[0], 3), dtype=np.ubyte)
                readImgsToList(x_imgs, imgs_N, imgsSeq, x_files)
                pm.printD("Saving on file: "+dirSeq+imgsSeqName+"_loaded")
                np.save(dirSeq+imgsSeqName+'_loaded', x_imgs, allow_pickle=False)
                pm.imshowI(x_imgs[0], "example")

            pm.printI(bcolors.OKGREEN+"Done: "+dirSeq+imgsSeqName+"_loaded"+bcolors.ENDC)

        if os.path.isfile(dirSeq+'pose_loaded.npy'):
            pm.printD("Already converted [poses/"+dirSeqName+".txt]!!")
            pm.printI(bcolors.OKGREEN+"Done: "+dirSeq+"pose_loaded"+bcolors.ENDC)
        else:
            pm.printD("Converting --> [poses/"+dirSeqName+".txt]")

            y_test = np.empty((imgs_N, 12), dtype=np.float16)
            fileName = path_poses+dirSeqName+'.txt'
            if os.path.isfile(path_poses+dirSeqName+'.txt'):
                readPosesFromFile(y_test, imgs_N, fileName)

                pm.printD("Saving on file: "+dirSeq+"pose_loaded")
                np.save(dirSeq+"pose_loaded", y_test, allow_pickle=False)

                pm.printI(bcolors.OKGREEN+"Done: "+dirSeq+"pose_loaded"+bcolors.ENDC)
            else:
                pm.printD(bcolors.WARNING+fileName+" does not exists!!"+bcolors.ENDC)



def main():
    pm.printI(bcolors.OKYELLOW+"Checking directories"+bcolors.ENDC+" ###\n")
    dirs = [dir_main, dir_Dataset, path_sequences, dir_Model, dir_History]
    checkExistDirs(dirs)
    pm.printI("Directories checked!\n")

    pm.printI(bcolors.OKYELLOW+"Converting dataset"+bcolors.ENDC+" ###")
    convertDataset()
    pm.printI("Done dataset convertion!\n")

if __name__ == "__main__":
    main()











