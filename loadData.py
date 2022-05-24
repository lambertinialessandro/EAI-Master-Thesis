
import os
import math
import numpy as np
import torch

import params
from EnumPreproc import EnumPreproc
from utility import PM, bcolors

# TODO
def isRotationMatrix(R):
    n_R = np.linalg.norm(
              np.identity(3, dtype = R.dtype) - np.dot(np.transpose(R), R)
          )
    return n_R < 1e-6
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
    return np.concatenate((p, angles))

class DataGeneretor():
    path_sequences = params.path_sequences
    path_poses = params.path_poses
    bachSize = params.BACH_SIZE
    numBatch = params.NUM_BACH
    numImgs4Iter = numBatch*bachSize
    step = params.STEP

    def __init__(self, sequence, imageDir, prepreocF, attach=False):
        self.sequence = sequence
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
        self.nameImgs = sorted(os.listdir(self.path2sequence))
        # load all the poses (light weight)
        self.loadedPoses = self._load_poses_init()
        # num of poses
        self.numImgs, _ = self.loadedPoses.shape
        # num of poses should be equal tu the number of images
        assert self.numImgs == len(self.nameImgs)

        #│self.numImgs = 15 * 3 # (int(sequence)+1)*self.bachSize + self.step # 15 # TODO  remove

        self.numBatchImgs = (self.numImgs - self.step)//self.numImgs4Iter

        # var for the iter
        self.currPos = 0
        self.maxPos = self.numBatchImgs#//(self.numImgs4Iter)
        ##self.currPos = self.maxPos

    def _load_poses_init(self):
        posesSet = []
        with open(self.path2pose, 'r') as f:
            for line in f:
                posef = np.fromstring(line, dtype=float, sep=' ')
                pose = poseFile2poseRobot(posef)
                posesSet.append(pose)
        return np.array(posesSet)

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

            nb = (diff-self.step) // self.bachSize
            if nb == 0:
                raise StopIteration
        else:
            # set of images with shape (numBatch, bachSize, ...)
            nb = self.numBatch
            appPos = self.currPos*self.numImgs4Iter

        imagesSet = self._load_images(appPos, nb)
        posesSet = self._load_poses(appPos, nb)
        pos = self.currPos
        self.currPos = self.currPos + 1

        if self.attach:
            imagesSet, posesSet = self._attach2Torch(imagesSet, posesSet)
        return imagesSet, posesSet, pos, nb

    def _load_images(self, pos, nb):
        imagesSet = []
        names = self.nameImgs[pos:pos+(nb*self.bachSize)+self.step]

        img1 = None
        img2 = None

        for i in range(nb):
            imagesSet.append([])
            for j in range(self.bachSize):
                name = names[i*self.bachSize+j]
                imgPath = os.path.join(self.path2sequence, name)
                img1 = self.prepreocF.processImage(imgPath)

                name = names[i*self.bachSize+j+self.step]
                imgPath = os.path.join(self.path2sequence, name)
                img2 = self.prepreocF.processImage(imgPath)

                h1, w1, c1 = img1.shape
                h2, w2, c2 = img2.shape
                assert h1 == h2 and w1 == w2 and c1 == c2

                img = np.concatenate([img1, img2], axis=-1)
                imagesSet[i].append(img)

        return np.array(imagesSet)

    def _load_poses(self, pos, nb):
        posesSet = []
        pos1 = None
        pos2 = None

        for i in range(nb):
            posesSet.append([])
            for j in range(self.bachSize):
                pos1 = self.loadedPoses[pos+i*self.bachSize+j]
                pos2 = self.loadedPoses[pos+i*self.bachSize+j+self.step]

                pose = pos2-pos1
                posesSet[i].append(pose)
        return np.array(posesSet)

    def _attach2Torch(self, imagesSet, posesSet):
        imagesSet = [torch.FloatTensor(imagesSet).to(params.DEVICE)]
        posesSet = [torch.FloatTensor(posesSet).to(params.DEVICE)]

        nb, bb, hh, ww, cc = imagesSet[0].size()
        nb2, bb2, pp = posesSet[0].size()
        assert nb == nb2 and bb == bb2

        imagesSet = torch.stack(imagesSet).view(-1, bb, cc, ww, hh)
        posesSet = torch.stack(posesSet).view(-1, bb, pp)

        return imagesSet, posesSet

    def __str__(self):
        return f"sequence {self.sequence}\n"+\
               f"bachSize {self.bachSize}\n"+\
               f"numBatch {self.numBatch}\n"+\
               f"numImgs4Iter {self.numImgs4Iter}\n"+\
               f"step {self.step}\n"+\
               "\n"+\
               f"numImgs {self.numImgs}\n"+\
               f"numBatchImgs {self.numBatchImgs}\n"+\
               f"currPos {self.currPos}\n"+\
               f"maxPos {self.maxPos}\n"

class RandomDataGeneretor():
    path_sequences = params.path_sequences
    path_poses = params.path_poses
    bachSize = params.BACH_SIZE
    numBatch = params.NUM_BACH
    numImgs4Iter = numBatch*bachSize
    step = params.STEP
    iters = len(params.trainingSeries)* params.RDG_ITER

    def __init__(self, sequences, imageDir, prepreocF, attach=False):
        self.sequences = sequences
        self.attach = attach

        self.currPos = 0
        self.shiftPos = 0
        self.maxPos = 0

        self.dgToDo = []
        for s in sequences:
            dg = DataGeneretor(s, imageDir, prepreocF, attach=False)
            self.dgToDo.append(dg)
            self.maxPos = self.maxPos + dg.maxPos
        self.maxIters = self.maxPos//self.iters
        self.dgDone = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.currPos > self.maxIters:
            for dg in self.dgToDo:
                self.dgDone.append(dg)
                self.dgToDo.remove(dg)
            raise StopIteration
        elif self.currPos == self.maxIters:
            imagesSet = None
            posesSet = None
            seq = []
            nb = 0
            while len(self.dgToDo) > 0:
                try:
                    dgToDo_pos = (self.currPos-self.shiftPos)%len(self.dgToDo)
                    #print(dgToDo_pos)
                    #print(self.dgToDo[dgToDo_pos].sequence)
                    imageSet, poseSet, _, nb_dg = self.dgToDo[dgToDo_pos].__next__()
                    nb = nb + nb_dg

                    seq.append(self.dgToDo[dgToDo_pos].sequence)
                    if imagesSet is None:
                        imagesSet = imageSet
                        posesSet = poseSet
                    else:
                        imagesSet = np.append(imagesSet, imageSet, axis=0)
                        posesSet = np.append(posesSet, poseSet, axis=0)
                except StopIteration:
                    #"print(f"-- terminated {self.dgToDo[dgToDo_pos].sequence}")
                    self.shiftPos = 1
                    self.dgDone.append(self.dgToDo[dgToDo_pos])
                    self.dgToDo.remove(self.dgToDo[dgToDo_pos])
            if imagesSet is None or posesSet is None:
                raise StopIteration
        else:
            imagesSet = None
            posesSet = None
            seq = []
            nb = 0
            for i in range(self.iters):
                while True:
                    try:
                        dgToDo_pos = (self.currPos+i-self.shiftPos)%len(self.dgToDo)
                        #print(dgToDo_pos)
                        #print(self.dgToDo[dgToDo_pos].sequence)
                        imageSet, poseSet, _, nb_dg = self.dgToDo[dgToDo_pos].__next__()
                        nb = nb + nb_dg

                        seq.append(self.dgToDo[dgToDo_pos].sequence)
                        if imagesSet is None:
                            imagesSet = imageSet
                            posesSet = poseSet
                        else:
                            imagesSet = np.append(imagesSet, imageSet, axis=0)
                            posesSet = np.append(posesSet, poseSet, axis=0)
                        break
                    except StopIteration:
                        #"print(f"-- terminated {self.dgToDo[dgToDo_pos].sequence}")
                        self.shiftPos = 1
                        self.dgDone.append(self.dgToDo[dgToDo_pos])
                        self.dgToDo.remove(self.dgToDo[dgToDo_pos])

        if self.attach:
            imagesSet, posesSet = self._attach2Torch(imagesSet, posesSet)

        pos = self.currPos
        self.currPos = self.currPos + 1
        return imagesSet, posesSet, pos, seq, nb

    def _attach2Torch(self, imagesSet, posesSet):
        imagesSet = [torch.FloatTensor(imagesSet).to(params.DEVICE)]
        posesSet = [torch.FloatTensor(posesSet).to(params.DEVICE)]

        nb, bb, hh, ww, cc = imagesSet[0].size()
        nb2, bb2, pp = posesSet[0].size()
        assert nb == nb2 and bb == bb2

        imagesSet = torch.stack(imagesSet).view(-1, bb, cc, ww, hh)
        posesSet = torch.stack(posesSet).view(-1, bb, pp)

        return imagesSet, posesSet

    def __str__(self):
        return f"sequences {self.sequences}\n"+\
               f"maxPos {self.maxPos}\n\n"+\
               "dgToDo: \n"+\
               ' '.join([f"{dg}\n\n" for dg in self.dgToDo])+\
               "dgDone: \n"+\
               ' '.join([f"{dg}\n\n" for dg in self.dgDone])


########### TODO

import cv2
import glob
#import time

def getImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (params.WIDTH, params.HEIGHT), interpolation=cv2.INTER_LINEAR)
    return img

def loadImages(path, suffix):
    #numImgs = 0#len(os.listdir(path))
    #print("Path: ".format(path))
    #print("Num of imges {}".format(numImgs))

    #initT = time.time()
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

    #elapsedT = time.time() - initT
    #print("Time needed: %.2fs"%(elapsedT))
    imagesSet = np.reshape(imagesSet, (-1, params.CHANNELS, params.WIDTH, params.HEIGHT))
    return imagesSet

def loadPoses(path):
    #print("Path: ".format(path))

    suffix = "_pose_loaded.npy"
    #initT = time.time()
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
                pose2 = poseFile2poseRobot(matrix)

                if notFirstIter:
                    pose = pose2-pose1
                    posesSet.append(pose)
                else:
                    notFirstIter = True

                pose1 = pose2
            posesSet = np.array(posesSet)

    #elapsedT = time.time() - initT
    #print("Time needed: %.2fs"%(elapsedT))
    return posesSet

def attach2Torch(imagesSet, posesSet):
    imagesSet = [torch.FloatTensor(imagesSet).to(params.DEVICE)] #[0:100]
    posesSet = [torch.FloatTensor(posesSet).to(params.DEVICE)] #[0:100]

    #print("Details of X :")
    #print(imagesSet[0].size())
    #print("Details of y :")
    #print(posesSet[0].size())

    imagesSet = torch.stack(imagesSet).view(-1, params.BACH_SIZE, params.CHANNELS,
                                            params.WIDTH, params.HEIGHT)
    posesSet = torch.stack(posesSet).view(-1, params.BACH_SIZE, params.NUM_POSES)
    #print("Details of X :")
    #print(imagesSet.size())
    #print("Details of y :")
    #print(posesSet.size())
    return imagesSet, posesSet

def DataLoader(datapath, attach=True, suffixType=1, sequence='00'):
  imgPath = os.path.join(datapath, 'sequences', sequence, 'image_2')
  posesPath = os.path.join(datapath, 'poses', sequence)

  if suffixType==1:
      suffix = "_{}_{}_loaded.npy".format(params.WIDTH, params.HEIGHT)
  elif suffixType==2:
      suffix = "_{}_{}_Quat_loaded.npy".format(params.WIDTH, params.HEIGHT)
  else:
      raise ValueError

  imagesSet = loadImages(imgPath, suffix)
  posesSet = loadPoses(posesPath)

  if attach:
    imagesSet, posesSet = attach2Torch(imagesSet, posesSet)

  return imagesSet, posesSet

########### TODO

def main():
    import gc
    import random

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




    rdg = RandomDataGeneretor(["00", "01", "03", "04"], imageDir, prepreocF, attach=False)
    print(rdg)

    for imageBatchSet, posesBatchSet, pos, nb in rdg:
        print(f"pos: {pos}")
        print(f"imageBatchSet: {imageBatchSet.shape}")
        print(f"posesBatchSet: {posesBatchSet.shape}")
        for imagesSet, posesSet in zip(imageBatchSet, posesBatchSet):
            print(f"imagesSet: {imagesSet.shape}")
            print(f"posesSet: {posesSet.shape}")
            for image, pose in zip(imagesSet, posesSet):
                print(pose)
                prepreocF.printImage(image[:, :, 0:3])
                prepreocF.printImage(image[:, :, 3:6])
                break
            break



    dataGens = []
    for s in params.trainingSeries:
        dataGens.append(DataGeneretor(s, imageDir, prepreocF, attach=False))

    while len(dataGens) > 0:
        pos = random.randint(0, len(dataGens)-1)

        try:
            imageBatchSet, posesBatchSet, pos, nb = dataGens[pos].__next__()

            print(f"pos: {pos}")
            print(f"nb: {nb}")
            print(f"imageBatchSet: {imageBatchSet.shape}")
            print(f"posesBatchSet: {posesBatchSet.shape}")
            for imagesSet, posesSet in zip(imageBatchSet, posesBatchSet):
                print(f"imagesSet: {imagesSet.shape}")
                print(f"posesSet: {posesSet.shape}")
                for image, pose in zip(imagesSet, posesSet):
                    #print(pose)
                    prepreocF.printImage(image[:, :, 0:3])
                    prepreocF.printImage(image[:, :, 3:6])
                    break
                break
        except StopIteration:
            dataGens.remove(dataGens[pos])


    return

    for dg in dataGens:
        print(f"numImgs: {dg.numImgs}")
        print(f"maxPos: {dg.maxPos}")
        print(f"numImgs4Iter: {dg.numImgs4Iter}")
        print(f"real num imgs: {dg.maxPos*dg.numImgs4Iter}")

        imageBatchSet, posesBatchSet, pos, nb = dg.__next__()
        print(f"pos: {pos}")
        print(f"imageBatchSet: {imageBatchSet.shape}")
        print(f"posesBatchSet: {posesBatchSet.shape}")
        for imagesSet, posesSet in zip(imageBatchSet, posesBatchSet):
            print(f"imagesSet: {imagesSet.shape}")
            print(f"posesSet: {posesSet.shape}")
            for image, pose in zip(imagesSet, posesSet):
                #print(pose)
                prepreocF.printImage(image[:, :, 0:3])
                prepreocF.printImage(image[:, :, 3:6])
                break
            break
    del dataGens
    gc.collect()

    dg = DataGeneretor(sequence, imageDir, prepreocF, attach=False)
    print(f"numImgs: {dg.numImgs}")
    print(f"maxPos: {dg.maxPos}")
    print(f"numImgs4Iter: {dg.numImgs4Iter}")
    print(f"real num imgs: {dg.maxPos*dg.numImgs4Iter}")

    try:
        for imageBatchSet, posesBatchSet, pos, nb in dg:
            print(f"pos: {pos}")
            print(f"imageBatchSet: {imageBatchSet.shape}")
            print(f"posesBatchSet: {posesBatchSet.shape}")
            for imagesSet, posesSet in zip(imageBatchSet, posesBatchSet):
                print(f"imagesSet: {imagesSet.shape}")
                print(f"posesSet: {posesSet.shape}")
                for image, pose in zip(imagesSet, posesSet):
                    #print(pose)
                    prepreocF.printImage(image[:, :, 0:3])
                    prepreocF.printImage(image[:, :, 3:6])
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
        for imageBatchSet, posesBatchSet, pos, nb in dg:
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


