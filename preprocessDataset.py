# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:17:53 2022

@author: lambe
"""


import os
import numpy as np
from enum import Enum

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dense, BatchNormalization

import cv2
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from utility import PrintManager

FLAG_DOWNLOAD_DATASET = False #@param {type:"boolean"}
FLAG_DEBUG_PRINT = True #@param {type:"boolean"}
FLAG_INFO_PRINT = True #@param {type:"boolean"}
pm = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT)


# global variables to save the tables/models
dir_main = 'C:/Users/lambe/OneDrive/Desktop/Dataset/'
#dir_main = 'E:/Università/Magistrale La Sapienza/2 Primo semestre/[EAI] AI for '+\
    #'VP in HCI & HRI/[EAI-VPH]/' #@param {type:"string"}

dir_Dataset = 'dataset/'#@param {type:"string"}
dir_Model = 'Model/'#@param {type:"string"}
dir_History = 'History/'#@param {type:"string"}

dir_Dataset = dir_main + dir_Dataset
dir_Model = dir_main + dir_Model
dir_History = dir_main + dir_History
dirs = [dir_main, dir_Dataset, dir_Model, dir_History]

img_size = (640, 174) # (1280,384) # (640, 174) = 111.360
path_dataset = dir_Dataset+'sequences/'


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
        pm.printProgressBar(pos, N)
        imgs[pos,:,:, :] = processImg(path+f, img_size)
        pos += 1
    pm.printProgressBar(N, N)

def convertDataset():
    for dirSeqName in os.listdir(path_dataset):
        dirSeq = path_dataset+dirSeqName+"/"
        if not os.path.isdir(dirSeq):
            continue

        for imgsSeqName in os.listdir(dirSeq):
            imgsSeq = dirSeq+imgsSeqName+"/"
            if not os.path.isdir(imgsSeq):
                continue

            if os.path.isfile(dirSeq+imgsSeqName+'_loaded.npy'):
                pm.printD("Already converted ["+dirSeqName+"/"+imgsSeqName+"]!!")
                continue
            else:
                pm.printD("Slow --> ["+dirSeqName+"/"+imgsSeqName+"]")
                x_files = sorted(os.listdir(imgsSeq))
                train_N = len(x_files)

                x_imgs = np.empty((train_N, img_size[1], img_size[0], 3), dtype=np.ubyte)
                readImgsToList(x_imgs, train_N, imgsSeq, x_files)
                pm.printD("Saving on file: "+dirSeq+imgsSeqName+"_loaded")
                np.save(dirSeq+imgsSeqName+'_loaded', x_imgs, allow_pickle=False)
                pm.imshowI(x_imgs[0], "example")


def main():
    checkExistDirs(dirs)

    convertDataset()

if __name__ == "__main__":
    main()











