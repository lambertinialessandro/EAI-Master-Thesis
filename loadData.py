
from enum import Enum
from abc import ABC, abstractmethod

import os
import numpy as np
import torch

from params import ParamsInstance as params

from modules.preprocess.PreprocessModule import PreprocessFactory
from modules.utility import PM, bcolors, poseFile2poseRobot


class AbstractDataGenerator(ABC):
    def __init__(self, sequence, imageDir, attach=False):
        self.sequence = sequence # sequence number
        self.imageDir = imageDir # dir between image_2 and image_3
        self.attach = attach #  if to attach to torch

        self.path2Sequences = os.path.join(params.path_sequences, sequence, imageDir) # input path
        self.path2Poses = os.path.join(params.path_poses, sequence + ".txt") # output path
        self.bachSize = params.BACH_SIZE # num of images
        self.step = params.STEP # images step

        # check dirs existance
        assert os.path.isdir(self.path2Sequences)
        assert os.path.isfile(self.path2Poses)

        # load all the poses (light weight)
        self.loadedPoses = self._load_poses_init()
        # num of poses
        self.numPoses, _ = self.loadedPoses.shape

        self.numBatchImgs = (self.numPoses - self.step)//self.bachSize

        # var for the iter
        self.currPos = 0
        self.maxPos = self.numBatchImgs

    def _load_poses_init(self):
        posesSet = []
        with open(self.path2Poses, 'r') as f:
            for line in f:
                posef = np.fromstring(line, dtype=float, sep=' ')
                pose = poseFile2poseRobot(posef)
                posesSet.append(pose)
        return np.array(posesSet)

    def _load_poses(self, pos, nb):
        posesSet = []
        pos1 = None
        pos2 = None

        for i in range(pos, pos + nb):
            pos1 = self.loadedPoses[i]
            pos2 = self.loadedPoses[i + self.step]

            pose = pos2-pos1
            posesSet.append(pose)
        return np.array(posesSet)

    def __iter__(self):
        return self

    def __next__(self):
        if self.currPos > self.maxPos:
            raise StopIteration
        elif self.currPos == self.maxPos:
            appPos = self.currPos*self.bachSize
            diff = self.numImgs - appPos
            if diff <= 0:
                raise StopIteration

            nb = (diff-self.step) // self.bachSize
            if nb == 0:
                raise StopIteration
        else:
            nb = self.bachSize
            appPos = self.currPos*self.bachSize

        imagesSet = self._load_images(appPos, nb)
        posesSet = self._load_poses(appPos, nb)

        pos = self.currPos
        self.currPos = self.currPos + 1

        if self.attach:
            imagesSet, posesSet = self._attach2Torch(imagesSet, posesSet)
        return imagesSet, posesSet, pos, nb

    @abstractmethod
    def _load_images(self, pos, nb):
        pass

    def _attach2Torch(self, imagesSet, posesSet):
        imagesSet = torch.FloatTensor(imagesSet).to(params.DEVICE)
        posesSet = torch.FloatTensor(posesSet).to(params.DEVICE)

        bb, hh, ww, cc = imagesSet.size()
        bb2, pp = posesSet.size()
        assert  bb == bb2

        return imagesSet, posesSet

    def __str__(self):
        return bcolors.LIGHTYELLOW+f"sequence {self.sequence}\n"+bcolors.ENDC+\
               f"bachSize {self.bachSize}\n"+\
               f"step {self.step}\n"+\
               "\n"+\
               f"numBatchImgs {self.numBatchImgs}\n"+\
               f"currPos {self.currPos}\n"+\
               f"maxPos {self.maxPos}\n"


class DataGeneretorPreprocessed(AbstractDataGenerator):
    def __init__(self, prepreocF, sequence, imageDir, attach=False):
        super().__init__(sequence, imageDir, attach=attach)

        self.prepreocF = prepreocF

        # function to process the image
        self.suffix =  self.prepreocF.suffix()
        self.numImgs = np.load(self.path2Sequences + self.suffix, allow_pickle=False).shape[0]

    def _load_images(self, pos, nb):
        imagesSet = np.load(self.path2Sequences + self.suffix, allow_pickle=False)
        imagesSet = np.reshape(imagesSet, (-1, params.CHANNELS, params.WIDTH, params.HEIGHT))
        imagesSet = np.reshape(imagesSet[pos:pos+nb],\
                               (self.bachSize, params.CHANNELS, params.WIDTH, params.HEIGHT))
        return imagesSet

    def __str__(self):
        return super().__str__()+\
               f"numImgs {self.numImgs}\n"


class DataGeneretorOnline(AbstractDataGenerator):
    def __init__(self, prepreocF, sequence, imageDir, attach=False):
        super().__init__(sequence, imageDir, attach=attach)

        self.prepreocF = prepreocF # function to process the image

        self.nameImgs = sorted(os.listdir(self.path2Sequences)) # names of all the images
        self.numImgs = len(self.nameImgs) # number of images

        assert self.numPoses == self.numImgs

    def _load_images(self, pos, nb):
        imagesSet = []

        img1 = None
        img2 = None

        for i in range(pos, pos + nb):

            imgPath = os.path.join(self.path2Sequences, self.nameImgs[i])
            img1 = self.prepreocF.processImage(imgPath)

            imgPath = os.path.join(self.path2Sequences, self.nameImgs[i+self.step])
            img2 = self.prepreocF.processImage(imgPath)

            img = np.concatenate([img1, img2], axis=-1)
            img = np.moveaxis(img, 2, 0)
            imagesSet.append(img)

        return np.array(imagesSet)

    def __str__(self):
        return super().__str__()+\
               f"numImgs {self.numImgs}\n"


class GeneratorType(Enum):
    PREPROCESS = "PREPROCESS"
    ONLINE = "ONLINE"

class DataGeneratorFactory():
    GeneratorType = GeneratorType

    @staticmethod
    def build(type_dg: GeneratorType,
              prepreocF, sequence, imageDir, attach=False):
        if type_dg == GeneratorType.PREPROCESS:
            dg = DataGeneretorPreprocessed(prepreocF, sequence, imageDir, attach=attach)
        elif type_dg == GeneratorType.ONLINE:
            dg = DataGeneretorOnline(prepreocF, sequence, imageDir, attach=attach)

        else:
            raise ValueError

        return dg


class RandomDataGeneretor():
    GeneratorType = GeneratorType

    def __init__(self, type_dg:GeneratorType,
                 prepreocF, sequences, imageDir, attach=False):
        self.prepreocF = prepreocF
        self.sequences = sequences
        self.imageDir = imageDir
        self.attach = attach

        self.path_sequences = params.path_sequences
        self.path_poses = params.path_poses
        self.bachSize = params.BACH_SIZE
        self.step = params.STEP

        self.iters = len(sequences)


        self.currPos = 0
        self.shiftPos = 0
        self.maxPos = 0

        self.dgToDo = []
        for s in sequences:
            dg = DataGeneratorFactory.build(type_dg, prepreocF, s, imageDir, attach=False)

            self.dgToDo.append(dg)
            self.maxPos = self.maxPos + dg.maxPos
        self.dgDone = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.currPos > self.maxPos:
            # for dg in self.dgToDo:
            #     self.dgDone.append(dg)
            #     self.dgToDo.remove(dg)
            raise StopIteration
        elif self.currPos == self.maxPos:
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

                    for _ in range(nb_dg):
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

                        for _ in range(nb_dg):
                            seq.append(self.dgToDo[dgToDo_pos].sequence)

                        if imagesSet is None:
                            imagesSet = imageSet
                            posesSet = poseSet
                        else:
                            imagesSet = np.append(imagesSet, imageSet, axis=0)
                            posesSet = np.append(posesSet, poseSet, axis=0)
                        break
                    except ZeroDivisionError:
                        raise StopIteration
                    except StopIteration:
                        #"print(f"-- terminated {self.dgToDo[dgToDo_pos].sequence}")
                        self.shiftPos += 1
                        self.dgDone.append(self.dgToDo[dgToDo_pos])
                        self.dgToDo.remove(self.dgToDo[dgToDo_pos])

        if self.attach:
            imagesSet, posesSet = self._attach2Torch(imagesSet, posesSet)

        self.shiftPos = 0
        pos = self.currPos
        self.currPos = self.currPos + 1
        return imagesSet, posesSet, pos, seq, nb

    def _attach2Torch(self, imagesSet, posesSet):
        imagesSet = torch.FloatTensor(imagesSet).to(params.DEVICE)
        posesSet = torch.FloatTensor(posesSet).to(params.DEVICE)

        bb, hh, ww, cc = imagesSet.size()
        bb2, pp = posesSet.size()
        assert bb == bb2

        return imagesSet, posesSet

    def __str__(self):
        return bcolors.LIGHTYELLOW+"sequences "+bcolors.ENDC+f"{self.sequences}\n"+\
               f"maxPos {self.maxPos}\n\n"+\
               "dgToDo: \n"+\
               ''.join([f"{dg}\n\n" for dg in self.dgToDo])+\
               "dgDone: \n"+\
               ''.join([f"{dg}\n\n" for dg in self.dgDone])



if __name__ == "__main__":
    PM.setFlags(True, True, False)


    sequence = "02"
    imageDir = "image_2"
    imgSize = (params.WIDTH, params.HEIGHT)
    prepreocF = PreprocessFactory.build(PreprocessFactory.PreprocessEnum.RESIZED, imgSize, step=params.STEP)


    dgo = DataGeneretorOnline(prepreocF, sequence, imageDir, attach=False)

    #pb = PM.printProgressBarI(0, dgo.maxPos-1)

    try:
        for imageBatchSet, posesBatchSet, pos, nb in dgo:
            print(imageBatchSet.shape)
            print(posesBatchSet.shape)
            break
            #pb.update(pos)
    except KeyboardInterrupt:
        pass
    finally:
        pass

    dgo = DataGeneretorPreprocessed(prepreocF, sequence, imageDir, attach=False)

    #pb = PM.printProgressBarI(0, dgo.maxPos-1)

    try:
        for imageBatchSet, posesBatchSet, pos, nb in dgo:
            print(imageBatchSet.shape)
            print(posesBatchSet.shape)
            break
            #pb.update(pos)
    except KeyboardInterrupt:
        pass
    finally:
        pass

    sequences = params.trainingSeries

    dgo = RandomDataGeneretor(RandomDataGeneretor.GeneratorType.ONLINE,
                              prepreocF, sequences, imageDir, attach=False)

    #pb = PM.printProgressBarI(0, dgo.maxPos-1)

    try:
        for imageBatchSet, posesBatchSet, pos, seq, nb in dgo:
            print(imageBatchSet.shape)
            print(posesBatchSet.shape)
            print(seq)
            break
            #pb.update(pos)
    except KeyboardInterrupt:
        pass
    finally:
        pass
