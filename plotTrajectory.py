
import os
import re
import numpy as np
import matplotlib.pyplot as plt

import params
from modules.utility import PM, bcolors, poseFile2poseRobot
from modules.preprocess.PreprocessModule import PreprocessFactory
from loadData import DataGeneretorPreprocessed, DataGeneretorOnline, RandomDataGeneretor

for sequence in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
    path2pose = f"./Dataset/poses/{sequence}.txt"

    pts_yTest = np.array([[0, 0, 0, 0, 0, 0]])

    posesSet = []
    with open(path2pose, 'r') as f:
        for line in f:
            posef = np.fromstring(line, dtype=float, sep=' ')
            pts_yTest = np.append(pts_yTest, [poseFile2poseRobot(posef)], axis=0)


    plt.plot(pts_yTest[:, 0], pts_yTest[:, 2], color='blue')
    plt.legend(['out', 'yTest'])
    plt.show()
