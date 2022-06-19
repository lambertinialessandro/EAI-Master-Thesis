
"""
r11 r12 r13 tx
r21 r22 r23 ty
r31 r32 r33 tz
0   0   0   1
is represented in the file as a single row:
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
"""

import os
import time
import numpy as np

import modules.utility as utility
from modules.utility import PM, bcolors


def readImgsToList(path, files, N, typePreproc):
    pos = 0
    img1 = []
    img2 = []
    imagesSet = []
    h1, w1, c1 = 0, 0, 0

    for f in files:
        PM.printProgressBarI(pos, N)
        img2 = typePreproc.processImage(os.path.join(path, f))

        if pos > 0:
            h1, w1, c1 = img1.shape
            h2, w2, c2 = img1.shape
            assert h1 == h2 and w1 == w2 and c1 == c2

            img = np.concatenate([img1, img2], axis=-1)
            imagesSet.append(img)

        img1 = img2
        pos += 1

    PM.printProgressBarI(N, N)
    return np.reshape(imagesSet, (-1, w1, h1, c1*2))


def readPosesFromFile(posesSet, N, path):
    pose1 = []
    pose2 = []

    with open(path, 'r') as f:
        for pos in range(N):
            PM.printProgressBarI(pos, N)
            posef = np.fromstring(f.readline(), dtype=float, sep=' ')
            pose2 = utility.poseFile2poseRobot(posef)

            if pos > 0:
                pose = pose2-pose1
                posesSet.append(pose)

            pose1 = pose2
        PM.printProgressBarI(N, N)


def convertDataset(path_sequences, path_poses,
                   listSequences, listDirs, listTypePreproc):
    PM.printI(bcolors.DARKYELLOW+"Converting dataset"+bcolors.ENDC+" ###", head="\n")


    for dirSeqName in listSequences:
        dirSeq = os.path.join(path_sequences, dirSeqName)
        if not os.path.isdir(dirSeq):
            continue

        PM.printI(bcolors.DARKYELLOW+"Converting: "+dirSeqName+bcolors.ENDC, head="\n")
        for imgsSeqName in listDirs:
            imgsSeq = os.path.join(dirSeq, imgsSeqName)
            if not os.path.isdir(imgsSeq):
                continue

            x_files = sorted(os.listdir(imgsSeq))
            imgs_N = len(x_files)

            for typePreproc in listTypePreproc:
                pathFinalFile = os.path.join(dirSeq, imgsSeqName + typePreproc.suffix())

                if os.path.isfile(pathFinalFile):
                    PM.printD("Already converted ["+pathFinalFile+"]!!")
                else:
                    PM.printD("Converting --> ["+pathFinalFile+"]")
                    initT = time.time()

                    imagesSet = readImgsToList(imgsSeq, x_files, imgs_N, typePreproc)
                    PM.printD("Saving on file: "+pathFinalFile)
                    np.save(pathFinalFile, imagesSet, allow_pickle=False)
                    elapsedT = time.time() - initT
                    PM.printD("Time needed: %.2fs for %d images"%(elapsedT, imgs_N))

                PM.printI(bcolors.DARKGREEN+"Done: "+pathFinalFile+bcolors.ENDC)


        poseFileName = os.path.join(path_poses, dirSeqName+"_pose_loaded.npy")
        if os.path.isfile(poseFileName):
            PM.printD("Already converted [poses/"+dirSeqName+".txt]!!")
            PM.printI(bcolors.DARKGREEN+"Done: "+poseFileName+bcolors.ENDC)
        else:
            PM.printD("Converting --> [poses/"+dirSeqName+".txt]")
            initT = time.time()

            posesSet = []
            fileName = os.path.join(path_poses, dirSeqName+'.txt')
            if os.path.isfile(fileName):
                readPosesFromFile(posesSet, imgs_N, fileName)

                PM.printD("Saving on file: "+poseFileName)
                np.save(poseFileName, posesSet, allow_pickle=False)
                elapsedT = time.time() - initT
                PM.printD("Time needed: %.2fs for %d poses"%(elapsedT, imgs_N))

                PM.printI(bcolors.DARKGREEN+"Done: "+poseFileName+bcolors.ENDC)
            else:
                PM.printD(bcolors.WARNING+fileName+" does not exists!!"+bcolors.ENDC)

    PM.printI("Done dataset convertion!")



if __name__ == "__main__":
    import params
    from modules.preprocess.PreprocessModule import PreprocessFactory

    PM.setFlags(True, True, False)


    path_sequences = params.path_sequences
    path_poses = params.path_poses

    listSequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        # ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']  os.listdir(path_sequences):
    listDirs = ['image_2'] # ['image_2']  os.listdir(dirSeq)
    listTypePreproc = [PreprocessFactory.build(
                            PreprocessFactory.PreprocessEnum.SOBEL,
                            (params.WIDTH, params.HEIGHT)),
                      ] # PreprocessFactory.listAllPreproc((params.WIDTH, params.HEIGHT))

    convertDataset(path_sequences, path_poses,
                    listSequences, listDirs, listTypePreproc)


