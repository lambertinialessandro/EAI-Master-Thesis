# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:23:56 2022

@author: lambe
"""

import os
import math
import numpy as np
import torch

import params
from EnumPreproc import EnumPreproc
from utility import PM, bcolors

# TODO
def isRotationMatrix(R):
    RT = np.transpose(R)
    n = np.linalg.norm(np.identity(3, dtype = R.dtype) - np.dot(RT, R))
    return n < 1e-6
# TODO
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
# TODO
def poseFile2poseRobot(posef):
    p = np.array([posef[3], posef[7], posef[11]])
    R = np.array([[posef[0], posef[1], posef[2]],
                  [posef[4], posef[5], posef[6]],
                  [posef[8], posef[9], posef[10]]])

    angles = rotationMatrix2EulerAngles(R)
    pose = np.concatenate((p, angles))
    return pose

class DataGeneretor():
    path_sequences = params.path_sequences
    path_poses = params.path_poses
    bachSize = params.BACH_SIZE
    numBatch = params.NUM_BACH
    numImgs4Iter = numBatch*bachSize
    step = params.BACH_SIZE

    def __init__(self, sequence, imageDir, prepreocF, attach=False):
        # str: path to sequences and poses
        self.path2sequence = os.path.join(self.path_sequences, sequence, imageDir)
        self.path2pose = os.path.join(self.path_poses, sequence + ".txt")
        # function to process the image
        self.prepreocF = prepreocF
        # if attach to torch
        self.attach = attach

        # check dirs existance
        assert os.path.isdir(self.path2sequence)
        assert os.path.isfile(self.path2pose)

        # names of all the images
        self.nameImgs = sorted(os.listdir(self.path2sequence))[0:71]
        # load all the poses (light weight)
        self.loadedPoses = self._load_poses()[0:71]
        # num of poses
        self.numImgs, _ = self.loadedPoses.shape
        # num of poses should be equal tu the number of images
        assert self.numImgs == len(self.nameImgs)
        self.numImgs = (self.numImgs - 1)//self.step

        # var for the iter
        self.currPos = 0
        self.maxPos = self.numImgs//(self.numImgs4Iter)

    def __iter__(self):
        return self

    def __next__(self):
        if self.currPos > self.maxPos:
            raise StopIteration
        elif self.currPos == self.maxPos:
            # set of images with shape (nb < numBatch, bachSize, ...)
            appPos = self.currPos*self.numImgs4Iter
            diff = self.numImgs - appPos
            if diff <= 0:
                raise StopIteration

            nb = diff // self.bachSize
            imagesSet = self._load_images(appPos, nb)
            posesSet = np.reshape(self.loadedPoses[appPos:appPos+diff],
                                  (nb, self.bachSize, params.NUM_POSES))
        else:
            # set of images with shape (numBatch, bachSize, ...)
            appPos = self.currPos*self.numImgs4Iter
            imagesSet = self._load_images(appPos, self.numBatch)
            posesSet = np.reshape(self.loadedPoses[appPos:appPos+self.numImgs4Iter],
                                  (self.numBatch, self.bachSize, params.NUM_POSES))

        pos = self.currPos
        self.currPos = self.currPos + 1

        if self.attach:
            imagesSet, posesSet = self._attach2Torch(imagesSet, posesSet)
        return imagesSet, posesSet, pos

    def _load_images(self, pos, nb):
        imagesSet = []
        names = self.nameImgs[pos:pos+self.numImgs4Iter]

        img1 = None
        imgPath = os.path.join(self.path2sequence, self.nameImgs[pos])
        img2 = self.prepreocF.processImage(imgPath)

        for i in range(nb):
            imagesSet.append([])
            for j in range(self.bachSize):
                name = names[i*self.bachSize+j]
                imgPath = os.path.join(self.path2sequence, name)

                img1 = img2
                img2 = self.prepreocF.processImage(imgPath)

                h1, w1, c1 = img1.shape
                h2, w2, c2 = img2.shape
                assert h1 == h2 and w1 == w2 and c1 == c2

                img = np.concatenate([img1, img2], axis=-1)
                imagesSet[i].append(img)

        return np.array(imagesSet)

    def _load_poses(self):
        posesSet = []
        with open(self.path2pose, 'r') as f:
            for line in f:
                posef = np.fromstring(line, dtype=float, sep=' ')
                pose = poseFile2poseRobot(posef)
                posesSet.append(pose)
        return np.array(posesSet)

    def _attach2Torch(self, imagesSet, posesSet):
        imagesSet = [torch.FloatTensor(imagesSet).to(params.DEVICE)] #[0:100]
        posesSet = [torch.FloatTensor(posesSet).to(params.DEVICE)] #[0:100]

        # print("Details of imagesSet :")
        # print(imagesSet[0].size())
        # print("Details of posesSet :")
        # print(posesSet[0].size())

        nb, bb, hh, ww, cc = imagesSet[0].size()
        nb2, bb2, pp = posesSet[0].size()
        assert nb == nb2 and bb == bb2

        imagesSet = torch.stack(imagesSet).view(-1, bb, cc, ww, hh)
        posesSet = torch.stack(posesSet).view(-1, bb, pp)

        # print("Details of imagesSet :")
        # print(imagesSet.size())
        # print("Details of posesSet :")
        # print(posesSet.size())

        return imagesSet, posesSet



def main():
    PM.printI(bcolors.DARKYELLOW+"Loading Data"+bcolors.ENDC+" ###")
    sequences = os.listdir(params.path_sequences)[0:11]
    num_seq = len(sequences)
    perc_train = 0.7
    num_train = round(num_seq * perc_train)
    num_test = num_seq - num_train

    PM.printI("Num seq Tot: {}".format(num_seq))
    PM.printI("Num seq Train: {} ({}%)".format(num_train, round(perc_train*100)))
    PM.printI("Train sequences: {}".format(params.trainingSeries))
    PM.printI("Num seq Test: {} ({}%)".format(num_test, round((1-perc_train)*100)))
    PM.printI("Test sequences: {}\n".format(params.testingSeries))

    sequence = "04"
    imageDir = "image_2"
    prepreocF = EnumPreproc.UNCHANGED((params.WIDTH, params.HEIGHT))
    # prepreocF = EnumPreproc.QUAD_CED((params.WIDTH, params.HEIGHT))
    # prepreocF = EnumPreproc.UNCHANGED((params.WIDTH, params.HEIGHT))

    dg = DataGeneretor(sequence, imageDir, prepreocF, attach=False)
    print(f"numImgs: {dg.numImgs}")
    print(f"maxPos: {dg.maxPos}")
    print(f"numImgs4Iter: {dg.numImgs4Iter}")
    print(f"real num imgs: {dg.maxPos*dg.numImgs4Iter}")

    try:
        for imageBatchSet, posesBatchSet, pos in dg:
            print(f"pos: {pos}")
            print(f"imageBatchSet: {imageBatchSet.shape}")
            print(f"posesBatchSet: {posesBatchSet.shape}")
            for imagesSet, posesSet in zip(imageBatchSet, posesBatchSet):
                print(f"imagesSet: {imagesSet.shape}")
                print(f"posesSet: {posesSet.shape}")
                for image, pose in zip(imagesSet, posesSet):
                    #print(pose)
                    prepreocF.printImage(image[:, :, 0:3])
                    break
                break
            # break
    except KeyboardInterrupt:
        pass
    finally:
        pass

    dg = DataGeneretor(sequence, imageDir, prepreocF, attach=True)
    print(f"numImgs: {dg.numImgs}")
    print(f"maxPos: {dg.maxPos}")
    print(f"numImgs4Iter: {dg.numImgs4Iter}")
    print(f"real num imgs: {dg.maxPos*dg.numImgs4Iter}")

    try:
        for imageBatchSet, posesBatchSet, pos in dg:
            print(f"pos: {pos}")
            print(f"imageBatchSet: {imageBatchSet.shape}")
            print(f"posesBatchSet: {posesBatchSet.shape}")
            for imagesSet, posesSet in zip(imageBatchSet, posesBatchSet):
                print(f"imagesSet: {imagesSet.shape}")
                print(f"posesSet: {posesSet.shape}")
                for image, pose in zip(imagesSet, posesSet):
                    #print(pose)
                    #prepreocF.printImage(image[:, :, 0:3])
                    break
                break
            # break
    except KeyboardInterrupt:
        pass
    finally:
        pass

if __name__ == "__main__":
    main()
