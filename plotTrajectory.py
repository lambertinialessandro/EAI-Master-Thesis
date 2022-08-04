
import os
import re
import numpy as np
import matplotlib.pyplot as plt

import params
from modules.utility import PM, bcolors, poseFile2poseRobot
from modules.preprocess.PreprocessModule import PreprocessFactory
from loadData import DataGeneretorPreprocessed, DataGeneretorOnline, RandomDataGeneretor

for sequence in params.trainingSeries:
    path2pose = f"./Dataset/poses/{sequence}.txt"

    pts_yTest = np.array([[0, 0, 0, 0, 0, 0]])

    posesSet = []
    with open(path2pose, 'r') as f:
        posef = np.fromstring(f.readline(), dtype=float, sep=' ')
        pose1 = poseFile2poseRobot(posef)
        for line in f:
            pose2 = pose1
            posef = np.fromstring(line, dtype=float, sep=' ')
            pose1 = poseFile2poseRobot(posef)

            pose = pose2-pose1

            pts_yTest = np.append(pts_yTest, [pts_yTest[-1] + pose], axis=0)


    plt.plot(pts_yTest[:, 0], pts_yTest[:, 2], color='blue')
    plt.legend(['out', 'yTest'])
    plt.show()

    # ax = plt.axes(projection='3d')
    # ax.plot3D(pts_yTest[:, 0], pts_yTest[:, 1], pts_yTest[:, 2], color='blue')
    # plt.legend(['out', 'yTest'])
    # plt.show()