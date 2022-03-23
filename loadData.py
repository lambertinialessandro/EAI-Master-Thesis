# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:23:56 2022

@author: lambe
"""

import os
import random
import glob
import math
import numpy as np
import cv2
import torch

import params
from utility import PrintManager, bcolors

pm = PrintManager(params.FLAG_DEBUG_PRINT, params.FLAG_INFO_PRINT)


def splitSequences(sequences, num_train):
    x_train_seq = []
    x_test_seq = sequences.copy()

    for i in range(num_train):
        pos = random.randint(0, len(x_test_seq)-1)
        x_train_seq.append(x_test_seq.pop(pos))
    x_train_seq.sort()

    return x_train_seq, x_test_seq

def loadData(seq):
    N = len(seq)
    x = []
    y = []

    for pos in range(N):
        pm.printProgressBarI(pos, N)
        curSeq = seq[pos]+"/"
        imgsFileName = "image_2"
        x.append(np.load(params.path_sequences+curSeq+imgsFileName+"_loaded.npy", allow_pickle=False))
        y.append(np.load(params.path_sequences+curSeq+"pose_loaded.npy", allow_pickle=False))
    pm.printProgressBarI(N, N)

    return x, y

def getImage(path):
  img = cv2.imread(path)
  img = cv2.resize(img, (params.WIDTH, params.HEIGHT), interpolation=cv2.INTER_LINEAR)
  return img

def loadImages(path, suffix):
  if os.path.isfile(path + suffix):
    imagesSet = np.load(path + suffix, allow_pickle=False)
    print(imagesSet.shape)
  else:
    notFirstIter = False
    img1 = []
    img2 = []
    imagesSet = []
    for img in glob.glob(path+'/*'):
      img2 = getImage(img)

      if notFirstIter:
        img = np.concatenate([img1, img2], axis=-1)
        imagesSet.append(img)
      else:
        notFirstIter = True

      img1 = img2
  imagesSet = np.reshape(imagesSet, (-1, params.CHANNELS, params.WIDTH, params.HEIGHT))
  return imagesSet

def isRotationMatrix(R):
  RT = np.transpose(R)
  n = np.linalg.norm(np.identity(3, dtype = R.dtype) - np.dot(RT, R))
  return n < 1e-6

def rotationMatrixToEulerAngles(R):
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

def matrix2pose(mat):
  p = np.array([mat[3], mat[7], mat[11]])
  R = np.array([[mat[0], mat[1], mat[2]],
                [mat[4], mat[5], mat[6]],
                [mat[8], mat[9], mat[10]]])

  angles = rotationMatrixToEulerAngles(R)
  pose = np.concatenate((p, angles))
  return pose

def loadPoses(path):
  suffix = "_pose_loaded.npy"
  if os.path.isfile(path + suffix):
    posesSet = np.load(path + suffix, allow_pickle=False)
  else:
    notFirstIter = False
    pose1 = []
    pose2 = []
    posesSet = []
    with open(path + ".txt", 'r') as f:
      lines = f.readlines()
      for line in lines:
        matrix = np.fromstring(line, dtype=float, sep=' ')
        pose2 = matrix2pose(matrix)

        if notFirstIter:
          pose = pose2-pose1
          posesSet.append(pose)
        else:
          notFirstIter = True

        pose1 = pose2
      posesSet = np.array(posesSet)
  return posesSet

def DataLoader(datapath, attach=True, suffixType=1, sequence='00'):
  imgPath = os.path.join(datapath, 'sequences', sequence, 'image_2')
  posesPath = os.path.join(datapath, 'poses', sequence)

  if suffixType==1:
      suffix = "_{}_{}_loaded.npy".format(params.WIDTH, params.HEIGHT)
  elif suffixType==2:
      suffix = "_{}_{}_Quat_loaded.npy".format(params.WIDTH, params.HEIGHT)
  else:
      raise ValueError

  if attach:
    imagesSet = [torch.FloatTensor(loadImages(imgPath, suffix)).to(params.DEVICE)] #[0:100]
    posesSet = [torch.FloatTensor(loadPoses(posesPath)).to(params.DEVICE)] #[0:100]

    print("Details of X :")
    print(imagesSet[0].size())
    print("Details of y :")
    print(posesSet[0].size())

    imagesSet = torch.stack(imagesSet).view(-1, params.BACH_SIZE, params.CHANNELS,
                                            params.WIDTH, params.HEIGHT)
    posesSet = torch.stack(posesSet).view(-1, params.BACH_SIZE, params.NUM_POSES)
    print("Details of X :")
    print(imagesSet.size())
    print("Details of y :")
    print(posesSet.size())
  else:
    imagesSet = loadImages(imgPath, suffix)
    posesSet = loadPoses(posesPath)

  return imagesSet, posesSet

def main():
    pm.printI(bcolors.DARKYELLOW+"Loading Data"+bcolors.ENDC+" ###")
    sequences = os.listdir(params.path_sequences)[0:2] # [0:11]
    num_seq = len(sequences)
    perc_train = 0.7
    num_train = round(num_seq * perc_train)
    num_test = num_seq - num_train
    train_seq, test_seq = splitSequences(sequences, num_train)
    pm.printI("Num seq Tot: {}".format(num_seq))
    pm.printI("Num seq Train: {} ({}%)".format(num_train, round(perc_train*100)))
    pm.printI("Train sequences: {}".format(train_seq))
    pm.printI("Num seq Test: {} ({}%)".format(num_test, round((1-perc_train)*100)))
    pm.printI("Test sequences: {}\n".format(test_seq))

    pm.printI(bcolors.DARKGREEN+"Loading Train data"+bcolors.ENDC+" ###")
    #x_train, y_train = loadData(train_seq)
    pm.printI("Loading Train data done!\n")

    pm.printI(bcolors.DARKGREEN+"Loading Test data"+bcolors.ENDC+" ###")
    #x_test, y_test = loadData(test_seq)
    pm.printI("Loading Test data done!\n")

    #return x_train, y_train, x_test, y_test



if __name__ == "__main__":
    main()
