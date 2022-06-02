
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

from EnumPreproc import EnumPreproc
import params
from utility import PM, bcolors


def checkExistDirs(dirs):
    for dir in dirs:
      if not os.path.exists(dir):
        os.makedirs(dir)
        PM.printD(dir + " --> CREATED")
      else:
        PM.printD(dir + " --> ALREADY EXIST")

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
            PM.printProgressBarI(pos, N)
            posef = np.fromstring(f.readline(), dtype=float, sep=' ')
            pose2 = poseFile2poseRobot(posef)

            if pos > 0:
                pose = pose2-pose1
                posesSet.append(pose)

            pose1 = pose2
        PM.printProgressBarI(N, N)

def convertDataset(listTypePreproc):
    for dirSeqName in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
        #os.listdir(path_sequences):
        dirSeq = os.path.join(params.path_sequences, dirSeqName)
        if not os.path.isdir(dirSeq):
            continue

        PM.printI(bcolors.DARKYELLOW+"Converting: "+dirSeqName+bcolors.ENDC, head="\n")
        for imgsSeqName in os.listdir(dirSeq): # ['image_2']
            imgsSeq = os.path.join(dirSeq, imgsSeqName)
            if not os.path.isdir(imgsSeq):
                continue

            x_files = sorted(os.listdir(imgsSeq))
            imgs_N = len(x_files)

            for typePreproc in listTypePreproc:
                suffix = "_{}_{}_{}_loaded.npy".format(typePreproc.name, params.WIDTH, params.HEIGHT)
                pathFinalFile = os.path.join(dirSeq, imgsSeqName + suffix)

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


        poseFileName = os.path.join(params.path_poses, dirSeqName+"_pose_loaded.npy")
        if os.path.isfile(poseFileName):
            PM.printD("Already converted [poses/"+dirSeqName+".txt]!!")
            PM.printI(bcolors.DARKGREEN+"Done: "+poseFileName+bcolors.ENDC)
        else:
            PM.printD("Converting --> [poses/"+dirSeqName+".txt]")
            initT = time.time()

            posesSet = []
            fileName = os.path.join(params.path_poses, dirSeqName+'.txt')
            if os.path.isfile(fileName):
                readPosesFromFile(posesSet, imgs_N, fileName)

                PM.printD("Saving on file: "+poseFileName)
                np.save(poseFileName, posesSet, allow_pickle=False)
                elapsedT = time.time() - initT
                PM.printD("Time needed: %.2fs for %d poses"%(elapsedT, imgs_N))

                PM.printI(bcolors.DARKGREEN+"Done: "+poseFileName+bcolors.ENDC)
            else:
                PM.printD(bcolors.WARNING+fileName+" does not exists!!"+bcolors.ENDC)



def main():
    PM.printI(bcolors.DARKYELLOW+"Checking directories"+bcolors.ENDC+" ###\n")
    dirs = [params.dir_main, params.dir_Dataset, params.path_sequences,
            params.dir_Model, params.dir_History]
    checkExistDirs(dirs)
    PM.printI("Directories checked!")

    # listTypePreproc = EnumPreproc.listAllPreproc((WIDTH, HEIGHT))
    # listTypePreproc = [EnumPreproc.UNCHANGED((params.WIDTH, params.HEIGHT)),
    #                    EnumPreproc.QUAD_PURE((params.WIDTH, params.HEIGHT))]
    listTypePreproc = [EnumPreproc.SOBEL((params.WIDTH, params.HEIGHT))]

    PM.printI(bcolors.DARKYELLOW+"Converting dataset"+bcolors.ENDC+" ###", head="\n")
    convertDataset(listTypePreproc)
    PM.printI("Done dataset convertion!")

if __name__ == "__main__":
    main()











