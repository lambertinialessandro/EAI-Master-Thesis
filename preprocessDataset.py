
import os
import time
import numpy as np

import modules.utility as utility
from modules.utility import PM, bcolors


def readImgsToList(path, files, typePreproc):
    imagesSet = []

    img1 = None
    img2 = None

    pb = PM.printProgressBarI(0, len(files) - params.STEP - 1)
    for pos in range(len(files) - params.STEP):
        name = files[pos]
        img1 = typePreproc.processImage(os.path.join(path, name))

        name = files[pos + params.STEP]
        img2 = typePreproc.processImage(os.path.join(path, name))

        h1, w1, c1 = img1.shape
        h2, w2, c2 = img2.shape
        assert h1 == h2 and w1 == w2 and c1 == c2

        img = np.concatenate([img1, img2], axis=-1)
        img = np.reshape(img, (-1, c1+c2, w1, h1))[0]
        imagesSet.append(img)
        pb.update(pos)
    return imagesSet


def readPosesFromFile(path):
    loadedPoses = []
    posesSet = []
    with open(path, 'r') as f:
        for line in f:
            posef = np.fromstring(line, dtype=float, sep=' ')
            pose = utility.poseFile2poseRobot(posef)
            loadedPoses.append(pose)

    pose1 = None
    pose2 = None
    pb = PM.printProgressBarI(0, len(loadedPoses) - params.STEP)
    for pos in range(len(loadedPoses) - params.STEP):
        pose1 = loadedPoses[pos]
        pose2 = loadedPoses[pos + params.STEP]

        pose = pose2-pose1
        posesSet.append(pose)
        pb.update(pos)


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

                    imagesSet = readImgsToList(imgsSeq, x_files, typePreproc)
                    PM.printD("Saving on file: "+pathFinalFile)
                    np.save(pathFinalFile, imagesSet, allow_pickle=False)
                    elapsedT = time.time() - initT
                    PM.printD("Time needed: %.2fs for %d images"%(elapsedT, imgs_N))

                PM.printI(bcolors.DARKGREEN+"Done: "+pathFinalFile+bcolors.ENDC)


        poseFileName = os.path.join(path_poses, dirSeqName+f"_pose_{params.STEP}_loaded.npy")
        if os.path.isfile(poseFileName):
            PM.printD("Already converted [poses/"+dirSeqName+".txt]!!")
            PM.printI(bcolors.DARKGREEN+"Done: "+poseFileName+bcolors.ENDC)
        else:
            PM.printD("Converting --> [poses/"+dirSeqName+".txt]")
            initT = time.time()

            fileName = os.path.join(path_poses, dirSeqName+'.txt')
            if os.path.isfile(fileName):
                posesSet = readPosesFromFile(fileName)

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
    listDirs = ['image_2']
    listTypePreproc = [PreprocessFactory.build(
                            PreprocessFactory.PreprocessEnum.SOBEL,
                            (params.WIDTH, params.HEIGHT)),
                      ] # PreprocessFactory.listAllPreproc((params.WIDTH, params.HEIGHT))

    convertDataset(path_sequences, path_poses,
                    listSequences, listDirs, listTypePreproc)


