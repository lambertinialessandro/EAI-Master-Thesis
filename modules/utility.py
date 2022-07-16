
import os
import math
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import HTML, display


def checkExistDirs(dirs):
    for dir in dirs:
      if not os.path.exists(dir):
        os.makedirs(dir)
        PM.printD(dir + " --> CREATED")
      else:
        PM.printD(dir + " --> ALREADY EXIST")


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


class STDOUT_holder:
    def __init__(self, value, max_v, *args, **kargs):
        super().__init__(*args, **kargs)
        self.max_v = max_v
        self.length = 40

        percent = ("{0:." + str(2) + "f}").format(100 * (value / float(self.max_v)))
        filledLength = int(self.length * value // self.max_v)
        bar = '█' * filledLength + '-' * (self.length - filledLength)
        print(f'\r### Info Progress:|{bar}| {percent}% Complete', end='\r')
        # Print New Line on Complete
        if value == self.max_v:
            print()

    def update(self, value):
        percent = ("{0:." + str(2) + "f}").format(100 * (value / float(self.max_v)))
        filledLength = int(self.length * value // self.max_v)
        bar = '█' * filledLength + '-' * (self.length - filledLength)
        print(f'\r### Info Progress:|{bar}| {percent}% Complete', end='\r')
        # Print New Line on Complete
        if value == self.max_v:
            print()


class HTML_holder:
    def __init__(self, value, max_v, *args, **kargs):
        super().__init__(*args, **kargs)
        self.max_v = max_v
        self.progress = display(
            HTML(f"<progress value='{value}' max='{self.max_v}', style='width: 50%' >\
                  {value}</progress>"), display_id=True)

    def update(self, value):
        self.progress.update(
            HTML(f"<progress value='{value}' max='{self.max_v}', style='width: 50%' >\
                 {value}</progress>"))


class PrintManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if PrintManager._instance is None:
            PrintManager._instance = super(PrintManager, cls).__new__(cls)
        return PrintManager._instance

    def __init__(self, FLAG_DEBUG_PRINT=False, FLAG_INFO_PRINT=True, FLAG_OUT_HTML=False):
        self.FLAG_DEBUG_PRINT = FLAG_DEBUG_PRINT
        self.FLAG_INFO_PRINT = FLAG_INFO_PRINT
        self.FLAG_OUT_HTML = FLAG_OUT_HTML

    def setFlags(self, value_FDP: bool, value_FIP: bool, value_FOH: bool):
        self.FLAG_DEBUG_PRINT = value_FDP
        self.FLAG_INFO_PRINT = value_FIP
        self.FLAG_OUT_HTML = value_FOH

    # Debug
    def printD(self, msg, head=""):
      if self.FLAG_DEBUG_PRINT:
        print(head + "### Debug: {}".format(msg))

    def imshowD(self, img, title=""):
      if self.FLAG_DEBUG_PRINT:
        plt.title("### Debug " + title)
        plt.imshow(img)
        plt.show()

    def printProgressBarI(self, value, max_v):
        if self.FLAG_INFO_PRINT:
            if self.FLAG_OUT_HTML:
                return HTML_holder(value, max_v)
            else:
                return STDOUT_holder(value, max_v)

    # Info
    def printI(self, msg, head=""):
      if self.FLAG_INFO_PRINT:
        print(head + "### Info: {}".format(msg))

    def imshowI(self, img, title=""):
      if self.FLAG_INFO_PRINT:
        plt.title("### Info " + title)
        plt.imshow(img)
        plt.show()


class bcolors:
    LIGHTRED = '\x1b[1;31;10m'
    LIGHTGREEN = '\x1b[1;32;10m'
    LIGHTYELLOW = '\x1b[1;33;10m'
    LIGHTCYAN = '\x1b[38;5;14m'

    DARKBLUE = '\x1b[1;30;44m'
    DARKYELLOW = '\x1b[1;33;40m'
    DARKGREEN = '\x1b[1;32;40m'

    WARNING = '\x1b[0;30;41m'
    FAIL = '\x1b[0;30;43m'

    ENDC = '\x1b[0m'

PM = PrintManager()


def main():
    import time

    FLAG_DEBUG_PRINT = True
    FLAG_INFO_PRINT = True
    FLAG_OUT_HTML = False

    pm = PrintManager(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT, FLAG_OUT_HTML)

    pm.printD(bcolors.LIGHTGREEN+"ciao"+bcolors.ENDC)
    pm.printI(bcolors.DARKGREEN+"ciao"+bcolors.ENDC)

    pb = pm.printProgressBarI(0, 5-1)
    for i in range(5):
        pb.update(i)
        time.sleep(0.3)

if __name__ == "__main__":
    main()